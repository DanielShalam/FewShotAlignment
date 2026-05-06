import json
import argparse
import os
from pathlib import Path
import copy
import yaml
import torch
from src.utils import setup_logger, set_seed, save_checkpoint, load_checkpoint

from src.datasets.base_dataset import build_dataset, build_loaders, DatasetWrapper
from src.model import FlowAdapter, MultiLabelFlowAdapter, GuidedVelocity
from src.engine import train_one_epoch, evaluate, evaluate_multilabel, build_feature_cache, evaluate_cached

def save_results(output_dir, metrics):
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

"""
OOD:
FSA:
python main.py --config configs/ablation.yaml --eval_only --resume output/ImageNet/ablation/seed_42/shots_16/model_best.pth dataset ImageNet ood_dataset ImageNetR shots 16 seed 42
python main.py --config configs/ablation.yaml --eval_only --resume output/ImageNet/ablation/seed_42/shots_16/model_best.pth dataset ImageNet ood_dataset ImageNetA shots 16 seed 42

MLP:
python main_mlp.py --config configs/ablation_noop.yaml dataset SUN397 shots 16 seed 42
python main_mlp.py --config configs/ablation_noop.yaml dataset SUN397 shots 4 seed 42

MLP + OP:
python main_mlp.py --config configs/ablation.yaml dataset SUN397 shots 16 seed 42
python main_mlp.py --config configs/ablation.yaml dataset SUN397 shots 4 seed 42

Residual MLP + op:
python main_mlp.py --config configs/res.yaml dataset SUN397 shots 16 seed 42 - next 16
python main_mlp.py --config configs/res.yaml dataset SUN397 shots 4 seed 42

"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    # Allow overriding config from CLI
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = get_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides (simple logic)
    if args.opts:
        if len(args.opts) % 2 != 0:
            raise ValueError("Override options must be key-value pairs (e.g., 'epochs 100 lr 0.001')")
        for i in range(0, len(args.opts), 2):
            k = args.opts[i].lstrip("-")
            v = args.opts[i+1]
            try:
                cfg[k] = yaml.safe_load(v)
            except Exception:
                cfg[k] = v

    # Setup
    if args.output_dir is None:
        cfg_name = args.config.split('/')[-1].split('.yaml')[0]
        args.output_dir = os.path.join("./output", cfg["dataset"], cfg_name, f"seed_{cfg['seed']}", f"shots_{cfg['shots']}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(f"Starting execution for seed {cfg['seed']} with config {args.config}")
    
    set_seed(cfg['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    ood_dataset_name = cfg.get("ood_dataset", None)
    use_ood_eval = isinstance(ood_dataset_name, str) and len(ood_dataset_name) > 0

    train_cfg = copy.deepcopy(cfg)
    dataset = build_dataset(train_cfg)
    if use_ood_eval:
        # OOD protocol: train/val always on ImageNet, test on selected ImageNet variant.
        train_cfg["dataset"] = "ImageNet"
        logger.info(f"OOD mode enabled: train/val dataset=ImageNet, test dataset={ood_dataset_name}")

        test_cfg = copy.deepcopy(train_cfg)
        test_cfg["dataset"] = ood_dataset_name
        ood_dataset = build_dataset(test_cfg)
        cfg["classnames"] = ood_dataset.classnames
        cfg["dataset"] = ood_dataset_name
    else:
        cfg["classnames"] = dataset.classnames
        cfg["dataset"] = train_cfg["dataset"]

    # Model
    if cfg["dataset"] != "VinDrCXR":
        model = FlowAdapter(cfg).to(device)
    else:
        model = MultiLabelFlowAdapter(cfg).to(device)   # multi-label loss and tuning

    # Data
    logger.info(f"Building {cfg['dataset']} dataloaders...")
    train_loader, train_val_loader, val_loader, test_loader = build_loaders(
        train_cfg, dataset, model.train_tfm, model.eval_tfm, return_train_eval=True)

    if use_ood_eval:
        test_loader = torch.utils.data.DataLoader(
            DatasetWrapper(test_cfg, ood_dataset.test, transform=model.eval_tfm, is_train=False),
            batch_size=test_cfg["batch_size"],
            num_workers=test_cfg["num_workers"],
            drop_last=False,
            pin_memory=True,
        )
        
    logger.info(
        f"Loader sizes: train_batches={len(train_loader)}, train_eval_batches={len(train_val_loader)}, "
        f"val_batches={len(val_loader) if val_loader is not None else 0}, test_batches={len(test_loader)}"
    )

    # Linear alignment matrix (OP). Auto-load if resuming, or create it
    if args.resume:
        saved_cfg = load_checkpoint(model, args.resume)
        # model.create_bank(train_val_loader)
        # Note: load_checkpoint loads state_dict, which includes OP.W
    else:
        # Create support bank (Proto calculation + OP fitting)
        multi_map = getattr(dataset, "multi_map", None)
        model.create_bank(train_val_loader, multi_map=multi_map)

        # Build image-feature caches once to skip the encoder forward on all
        # subsequent intermediate evals and the final hyperparameter-tuning
        # sweep. Safe because the image encoder is frozen during training.
        _cache_test_feats, _cache_test_labels = None, None
        _cache_val_feats, _cache_val_labels = None, None
        if cfg["dataset"] != "VinDrCXR":
            logger.info("Building test feature cache (image encoder is frozen)...")
            _cache_test_feats, _cache_test_labels = build_feature_cache(
                model, test_loader, device, desc="Caching test feats")
            if val_loader is not None:
                logger.info("Building val feature cache...")
                _cache_val_feats, _cache_val_labels = build_feature_cache(
                    model, val_loader, device, desc="Caching val feats")
            logger.info(
                f"Caches built: test={tuple(_cache_test_feats.shape)}, "
                f"val={tuple(_cache_val_feats.shape) if _cache_val_feats is not None else None}"
            )

    # Eval Only Mode
    if args.eval_only:
        acc = evaluate(model, test_loader, device, alpha=cfg['alpha'])
        print(f"Test Accuracy (Before tuning): {acc:.2f}%")
        val_acc, best_params = model.tune_hyperparameters(val_loader, device=device)
        acc_tuned = evaluate(model, test_loader, device, alpha=best_params[0], t_end=best_params[1])
        print(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")
        print(f"Test Accuracy (After tuning): {acc_tuned:.2f}%")
        out_metrics = {
            "Test Accuracy (Before tuning)": acc,
            "Test Accuracy (After tuning)": acc_tuned,
            "Best alpha": best_params[0],
            "Best timestep": best_params[1]
        }
        save_results(args.output_dir, out_metrics)
        return

    # Training
    adapter_params = list(model.adapter.parameters())
    if cfg["text_adapter"]:
        adapter_params += list(model.t_adapter.parameters())
    # If OP.W is trainable (cfg['op_trainable']=True), include it as a
    # separate param group with a configurable lower LR (defaults to cfg['lr']
    # if op_lr is unset).
    if cfg.get("op_trainable", False) and isinstance(getattr(model.OP, "W", None), torch.nn.Parameter):
        op_lr = float(cfg.get("op_lr", cfg['lr']))
        param_groups = [
            {"params": adapter_params, "lr": cfg['lr'], "weight_decay": cfg['wd']},
            {"params": [model.OP.W], "lr": op_lr, "weight_decay": cfg['wd']},
        ]
        print(f"[optim] trainable OP.W (shape={tuple(model.OP.W.shape)}) with lr={op_lr}")
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(adapter_params, lr=cfg['lr'], weight_decay=cfg['wd'])
    # Optional linear warmup before cosine decay.
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    warmup_start_factor = float(cfg.get("warmup_start_factor", 1e-3))
    if warmup_epochs > 0 and warmup_epochs < cfg['epochs']:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start_factor, end_factor=1.0,
            total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'] - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cosine], milestones=[warmup_epochs])
        logger.info(f"LR schedule: linear warmup {warmup_epochs} epochs (start_factor={warmup_start_factor}) -> cosine decay")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    best_acc = 0.0
    eval_freq = cfg.get('eval_freq', cfg['epochs']) # Default to only evaluating at the end if not specified
    train_multi_map = dataset.multi_map if cfg['dataset'] == "VinDrCXR" else None
    grad_clip = float(cfg.get('grad_clip', 0.0))
    if grad_clip > 0:
        logger.info(f"Gradient clipping enabled with max_norm={grad_clip} (L2 norm)")
    for epoch in range(1, cfg['epochs'] + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            device,
            multi_map=train_multi_map,
            grad_clip=grad_clip,
        )
        if scheduler is not None:
            scheduler.step()
        
        if epoch % eval_freq == 0:
            print(f"--- Fast Evaluation at Epoch {epoch} ---")
            if cfg['dataset'] != "VinDrCXR":
                if _cache_test_feats is not None:
                    acc = evaluate_cached(model, _cache_test_feats, _cache_test_labels,
                                          device, alpha=cfg['alpha'])
                else:
                    acc = evaluate(model, test_loader, device, alpha=cfg['alpha'])
                print(f"Test Accuracy: {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    # Save best model logic can go here if needed.
            else:
                acc = evaluate_multilabel(model, test_loader, dataset.multi_map, device, alpha=cfg['alpha'], t_end=0.8, solver="dopri5")
                print(f"Test Result: {acc}")

    # Save Config and OP automatically
    save_checkpoint(model, optimizer, scheduler, cfg, epoch, args.output_dir, is_best=True)

    if cfg['dataset'] != "VinDrCXR":
        if _cache_test_feats is not None:
            acc = evaluate_cached(model, _cache_test_feats, _cache_test_labels,
                                  device, alpha=cfg['alpha'])
        else:
            acc = evaluate(model, test_loader, device, alpha=cfg['alpha'])
        print(f"Test Accuracy (Before tuning): {acc:.2f}%")

        print(f"Hyparparams tuning...")
        val_acc, best_params = model.tune_hyperparameters(
            val_loader, device=device,
            cached_feats=_cache_val_feats, cached_labels=_cache_val_labels,
        )
        print(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")

        if _cache_test_feats is not None:
            acc_tuned = evaluate_cached(model, _cache_test_feats, _cache_test_labels,
                                        device, alpha=best_params[0], t_end=best_params[1])
        else:
            acc_tuned = evaluate(model, test_loader, device, alpha=best_params[0], t_end=best_params[1])
        print(f"Test Accuracy (After tuning): {acc_tuned:.2f}%")
        out_metrics = {
            "Test Accuracy (Before tuning)": acc,
            "Test Accuracy (After tuning)": acc_tuned,
            "Best alpha": best_params[0],
            "Best timestep": best_params[1]
        }
    else:
        multi_map = dataset.multi_map
        acc_m = evaluate_multilabel(model, test_loader, multi_map, device, alpha=cfg['alpha'])
        logger.info(f"Test Result (Before tuning): {acc_m}")

        logger.info(f"Hyparparams tuning...")
        val_acc, best_params = model.tune_hyperparameters(val_loader, multi_map=dataset.multi_map, device=device)
        logger.info(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")

        acc_tuned = evaluate_multilabel(model, test_loader, multi_map, device, alpha=best_params[0], t_end=best_params[1])
        logger.info(f"Test Result (After tuning): {acc_tuned}")
        out_metrics = {
            "Test AUC (Before tuning)": acc_m['macro_AUROC'],
            "Test AUPRC (Before tuning)": acc_m['macro_AUPRC'],
            "Test AUC (After tuning)": acc_tuned['macro_AUROC'],
            "Test AUPRC (After tuning)": acc_tuned['macro_AUPRC'],
            "Best alpha": best_params[0],
            "Best timestep": best_params[1]
        }

    save_results(args.output_dir, out_metrics)
    return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import logging
        logger = logging.getLogger("FlowAdapter")
        logger.exception("An error occurred during execution:")
        raise