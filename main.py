import argparse
import os
from pathlib import Path
import yaml
import torch
from src.utils import setup_logger, set_seed, save_checkpoint, load_checkpoint

from src.datasets.base_dataset import build_dataset, build_loaders
from src.model import FlowAdapter, MultiLabelFlowAdapter
from src.engine import train_one_epoch, evaluate, evaluate_multilabel


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
    dataset = build_dataset(cfg)
    cfg["classnames"] = dataset.classnames

    # Model
    if cfg["dataset"] in ("VinDrCXR", "DeepLoc2"):
        model = MultiLabelFlowAdapter(cfg).to(device)
    else:
        model = FlowAdapter(cfg).to(device)   # multi-label loss and tuning

    # Data
    logger.info(f"Building {cfg['dataset']} dataloaders...")
    train_loader, train_val_loader, val_loader, test_loader = build_loaders(
        cfg, dataset, model.train_tfm, model.eval_tfm, return_train_eval=True)
    logger.info(
        f"Loader sizes: train_batches={len(train_loader)}, train_eval_batches={len(train_val_loader)}, "
        f"val_batches={len(val_loader) if val_loader is not None else 0}, test_batches={len(test_loader)}"
    )

    # Linear alignment matrix (OP). Auto-load if resuming, or create it
    if args.resume:
        saved_cfg = load_checkpoint(model, args.resume)
        # Note: load_checkpoint loads state_dict, which includes OP.W
    else:
        # Create support bank (Proto calculation + OP fitting)
        model.create_bank(train_val_loader)

    # Eval Only Mode
    if args.eval_only:
        acc = evaluate(model, test_loader, device, alpha=cfg['alpha'], t_end=1.)
        logger.info(f"Test Accuracy (Before tuning): {acc:.2f}%")
        val_acc, best_params = model.tune_hyperparameters(val_loader, device=device)
        acc_tuned = evaluate(model, test_loader, device, alpha=best_params[0], t_end=best_params[1])
        logger.info(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")
        logger.info(f"Test Accuracy (After tuning): {acc_tuned:.2f}%")
        return

    # Training
    params = model.adapter.parameters() if not cfg["text_adapter"] \
        else list(model.adapter.parameters()) + list(model.t_adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    import copy
    # --- EMA shadow params (optional; cfg.ema_decay<=0 disables) ---
    ema_decay = float(cfg.get('ema_decay', 0.0))
    ema_modules = [model.adapter] + ([model.t_adapter] if cfg['text_adapter'] else [])
    ema_shadow = None
    _ema_update = None
    _ema_swap = None
    _ema_restore = None
    if ema_decay > 0.0:
        ema_shadow = [{n: p.detach().clone() for n, p in m.named_parameters()} for m in ema_modules]
        @torch.no_grad()
        def _ema_update():
            for m, shadow in zip(ema_modules, ema_shadow):
                for n, p in m.named_parameters():
                    shadow[n].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
        def _ema_swap():
            # swap module params with shadow; return backup for restoration
            backups = []
            for m, shadow in zip(ema_modules, ema_shadow):
                b = {}
                for n, p in m.named_parameters():
                    b[n] = p.detach().clone()
                    p.data.copy_(shadow[n])
                backups.append(b)
            return backups
        def _ema_restore(backups):
            for m, b in zip(ema_modules, backups):
                for n, p in m.named_parameters():
                    p.data.copy_(b[n])
    best_val = -1.0
    best_state = None
    eval_freq = cfg.get('eval_freq', cfg['epochs']) # Default to only evaluating at the end if not specified
    train_multi_map = dataset.multi_map if cfg['dataset'] in ('VinDrCXR', 'DeepLoc2') else None
    for epoch in range(1, cfg['epochs'] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, epoch, device, multi_map=train_multi_map,
                               after_step=_ema_update)
        if scheduler is not None:
            scheduler.step()

        if epoch % eval_freq == 0:
            print(f"--- Eval at Epoch {epoch} ---")
            _bk = _ema_swap() if _ema_swap is not None else None
            if cfg['dataset'] not in ('VinDrCXR', 'DeepLoc2'):
                # Validation-driven best checkpoint selection (no test leakage)
                t_ends = cfg.get('fast_eval_t_ends', [0.5, 1.0])
                if val_loader is not None:
                    val_accs = [evaluate(model, val_loader, device, alpha=cfg['alpha'], t_end=te,
                                         solver=cfg.get('fast_eval_solver', 'dopri5'),
                                         steps=cfg.get('fast_eval_steps', None)) for te in t_ends]
                    best_te_idx = max(range(len(t_ends)), key=lambda i: val_accs[i])
                    val_acc = val_accs[best_te_idx]
                    print("Val Accuracy: " + "  ".join(f"τ={te:.2f}:{a:.2f}" for te,a in zip(t_ends, val_accs)) + f"  -> best τ={t_ends[best_te_idx]:.2f} acc={val_acc:.2f}%")
                    if val_acc > best_val:
                        best_val = val_acc
                        best_state = {
                            'adapter': copy.deepcopy(model.adapter.state_dict()),
                            't_adapter': copy.deepcopy(model.t_adapter.state_dict()) if cfg['text_adapter'] else None,
                        }
                # Optional monitoring on test (not used for selection)
                test_accs = [evaluate(model, test_loader, device, alpha=cfg['alpha'], t_end=te,
                                      solver=cfg.get('fast_eval_solver', 'dopri5'),
                                      steps=cfg.get('fast_eval_steps', None)) for te in t_ends]
                print("Test Accuracy: " + "  ".join(f"τ={te:.2f}:{a:.2f}" for te,a in zip(t_ends, test_accs)))
            else:
                acc = evaluate_multilabel(model, test_loader, dataset.multi_map, device, alpha=cfg['alpha'])
                print(f"Test Result: {acc}")
            if _bk is not None:
                _ema_restore(_bk)

    # Restore best-val checkpoint before final eval/tune
    if best_state is not None:
        model.adapter.load_state_dict(best_state['adapter'])
        if cfg['text_adapter'] and best_state.get('t_adapter') is not None:
            model.t_adapter.load_state_dict(best_state['t_adapter'])
        print(f"Loaded best-val checkpoint (val acc = {best_val:.2f}%)")

    # Save Config and OP automatically
    save_checkpoint(model, optimizer, scheduler, cfg, epoch, args.output_dir, is_best=True)

    if cfg['dataset'] not in ('VinDrCXR', 'DeepLoc2'):
        acc = evaluate(model, test_loader, device, alpha=cfg['alpha'])
        print(f"Test Accuracy (Before tuning): {acc:.2f}%")

        print(f"Hyparparams tuning...")
        val_acc, best_params = model.tune_hyperparameters(val_loader, device=device)
        print(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")

        acc_tuned = evaluate(model, test_loader, device, alpha=best_params[0], t_end=best_params[1])
        print(f"Test Accuracy (After tuning): {acc_tuned:.2f}%")
    else:
        multi_map = dataset.multi_map
        acc = evaluate_multilabel(model, test_loader, multi_map, device, alpha=cfg['alpha'])
        logger.info(f"Test Result (Before tuning): {acc}")

        logger.info(f"Hyparparams tuning...")
        val_acc, best_params = model.tune_hyperparameters(val_loader, multi_map=dataset.multi_map, device=device)
        logger.info(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")

        acc_tuned = evaluate_multilabel(model, test_loader, multi_map, device, alpha=best_params[0], t_end=best_params[1])
        logger.info(f"Test Result (After tuning): {acc_tuned}")

    return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import logging
        logger = logging.getLogger("FlowAdapter")
        logger.exception("An error occurred during execution:")
        raise