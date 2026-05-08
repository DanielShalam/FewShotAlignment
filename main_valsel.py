"""main_valsel.py — training with val-based model selection.

Variant of main.py that tracks **validation accuracy** during training and
keeps the checkpoint with the highest val accuracy. At the end:
 1. Load the best-val checkpoint.
 2. Run (optional) hyparam tuning on val.
 3. Report final test accuracy.

Used for BiomedCoOp replication where the original runs used val-based
selection (not fixed-length training).
"""
import argparse
import copy
import os
from pathlib import Path

import torch
import yaml

from src.datasets.base_dataset import build_dataset, DatasetWrapper, build_loaders
from src.engine import (
    build_feature_cache, evaluate, evaluate_cached, train_one_epoch,
)
from src.model import FlowAdapter
from src.utils import (
    load_checkpoint, save_checkpoint, set_seed, setup_logger,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def load_cfg(config_path, opts):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if opts:
        if len(opts) % 2 != 0:
            raise ValueError("Override opts must be key-value pairs")
        for i in range(0, len(opts), 2):
            k = opts[i].lstrip("-")
            v = opts[i + 1]
            try:
                cfg[k] = yaml.safe_load(v)
            except Exception:
                cfg[k] = v
    return cfg


def main():
    args = parse_args()
    cfg = load_cfg(args.config, args.opts)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(f"Starting execution for seed {cfg['seed']} with config {args.config}")
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataset + loaders
    dataset = build_dataset(cfg)
    cfg["classnames"] = dataset.classnames
    train_cfg = copy.deepcopy(cfg)

    # Model
    model = FlowAdapter(cfg).to(device)

    train_loader, train_val_loader, val_loader, test_loader = build_loaders(
        train_cfg, dataset, model.train_tfm, model.eval_tfm, return_train_eval=True,
    )
    logger.info(
        f"Loader sizes: train={len(train_loader)}, train_eval={len(train_val_loader)}, "
        f"val={len(val_loader) if val_loader is not None else 0}, test={len(test_loader)}"
    )
    if val_loader is None:
        raise ValueError("main_valsel.py requires a val loader; dataset has no val split")

    # OP + bank
    model.create_bank(train_val_loader)

    # Cache val + test features for fast eval
    logger.info("Building feature caches...")
    _val_feats, _val_labels = build_feature_cache(model, val_loader, device, desc="Caching val feats")
    _test_feats, _test_labels = build_feature_cache(model, test_loader, device, desc="Caching test feats")

    # Optimizer + scheduler (mirrors main.py)
    adapter_params = list(model.adapter.parameters())
    if cfg["text_adapter"]:
        adapter_params += list(model.t_adapter.parameters())
    if cfg.get("op_trainable", False) and isinstance(getattr(model.OP, "W", None), torch.nn.Parameter):
        op_lr = float(cfg.get("op_lr", cfg['lr']))
        param_groups = [
            {"params": adapter_params, "lr": cfg['lr'], "weight_decay": cfg['wd']},
            {"params": [model.OP.W], "lr": op_lr, "weight_decay": cfg['wd']},
        ]
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(adapter_params, lr=cfg['lr'], weight_decay=cfg['wd'])

    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    if warmup_epochs > 0 and warmup_epochs < cfg['epochs']:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=float(cfg.get("warmup_start_factor", 1e-3)),
            end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'] - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # Training with val-based checkpoint selection (multi-t_end grid as in old runs)
    eval_freq = int(cfg.get("eval_freq", cfg['epochs']))
    alpha = float(cfg.get("alpha", 0.0))
    grad_clip = float(cfg.get("grad_clip", 0.0))
    fast_eval_t_ends = cfg.get("fast_eval_t_ends", [0.5, 1.0])
    if isinstance(fast_eval_t_ends, str):
        fast_eval_t_ends = [float(x) for x in fast_eval_t_ends.split(",")]
    fast_eval_solver = cfg.get("fast_eval_solver", "dopri5")
    fast_eval_steps = cfg.get("fast_eval_steps", None)

    best_val_acc = -1.0
    best_epoch = 0
    best_te = None
    best_state = None  # full model state_dict + OP.W
    for epoch in range(1, cfg['epochs'] + 1):
        train_one_epoch(
            model, train_loader, optimizer, epoch, device,
            multi_map=None, grad_clip=grad_clip,
        )
        scheduler.step()

        if epoch % eval_freq == 0 or epoch == cfg['epochs']:
            # Validation across multiple t_ends; pick max.
            val_accs = [
                evaluate(model, val_loader, device, alpha=alpha, t_end=te,
                         solver=fast_eval_solver, steps=fast_eval_steps)
                for te in fast_eval_t_ends
            ]
            best_te_idx = max(range(len(fast_eval_t_ends)), key=lambda i: val_accs[i])
            val_acc = val_accs[best_te_idx]
            val_str = "  ".join(f"τ={te:.2f}:{a:.2f}" for te, a in zip(fast_eval_t_ends, val_accs))
            logger.info(f"ep {epoch:>3} Val {val_str}  -> best τ={fast_eval_t_ends[best_te_idx]:.2f} acc={val_acc:.2f}%")
            # Test (monitoring only — not used for selection)
            test_accs = [
                evaluate(model, test_loader, device, alpha=alpha, t_end=te,
                         solver=fast_eval_solver, steps=fast_eval_steps)
                for te in fast_eval_t_ends
            ]
            test_str = "  ".join(f"τ={te:.2f}:{a:.2f}" for te, a in zip(fast_eval_t_ends, test_accs))
            logger.info(f"ep {epoch:>3} Test {test_str}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_te = fast_eval_t_ends[best_te_idx]
                best_state = {
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "OP_W": model.OP.W.detach().cpu().clone() if model.OP.W is not None else None,
                }
                logger.info(f"  new best val={best_val_acc:.2f}% @ epoch {best_epoch}, τ={best_te}")

    # Load best val checkpoint (full model state)
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"], strict=False)
        if best_state["OP_W"] is not None:
            model.OP.W = best_state["OP_W"].to(device)
        logger.info(f"Loaded best-val checkpoint (val={best_val_acc:.2f}% @ ep {best_epoch}, τ={best_te})")

    # Before tuning eval
    test_acc_before = evaluate_cached(model, _test_feats, _test_labels, device, alpha=alpha)
    logger.info(f"Test Accuracy (Before tuning): {test_acc_before:.2f}%")

    # Hyparam tuning over (alpha, t_end) on val
    logger.info("Hyparparams tuning...")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t_ends = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    best_hp_val = -1.0
    best_alpha, best_t_end = alpha, 0.5
    for a in alphas:
        for t in t_ends:
            v = evaluate_cached(model, _val_feats, _val_labels, device, alpha=a, t_end=t)
            if v > best_hp_val:
                best_hp_val = v
                best_alpha, best_t_end = a, t
    logger.info(f"Best Hyparparams: alpha={best_alpha}, timestep={best_t_end}  val={best_hp_val:.2f}%")

    test_acc_after = evaluate_cached(
        model, _test_feats, _test_labels, device, alpha=best_alpha, t_end=best_t_end)
    logger.info(f"Test Accuracy (After tuning): {test_acc_after:.2f}%")

    # Save final (best-val) checkpoint
    save_checkpoint(model, optimizer, scheduler, cfg, best_epoch, args.output_dir, is_best=True)
    import json
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "Test Accuracy (Before tuning)": test_acc_before,
            "Test Accuracy (After tuning)": test_acc_after,
            "Best alpha": best_alpha,
            "Best timestep": best_t_end,
        }, f, indent=2)


if __name__ == "__main__":
    main()
