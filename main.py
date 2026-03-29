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
        # Append seed to output directory to prevent overlapping in parallel runs
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
    if cfg["dataset"] != "VinDrCXR":
        model = FlowAdapter(cfg).to(device)
    else:
        model = MultiLabelFlowAdapter(cfg).to(device)   # multi-label loss and tuning

    # Data
    logger.info(f"Building {cfg['dataset']} dataloaders...")
    train_loader, train_val_loader, val_loader, test_loader = build_loaders(
        cfg, dataset, model.train_tfm, model.eval_tfm, return_train_eval=True)

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
        val_acc, best_params = model.tune_hyperparameters(train_val_loader, device=device)
        acc_tuned = evaluate(model, test_loader, device, alpha=best_params[0], t_end=best_params[1])
        logger.info(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")
        logger.info(f"Test Accuracy (After tuning): {acc_tuned:.2f}%")
        return

    # Training
    params = model.adapter.parameters() if not cfg["text_adapter"] \
        else list(model.adapter.parameters()) + list(model.t_adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    best_acc = 0.0
    eval_freq = cfg.get('eval_freq', cfg['epochs']) # Default to only evaluating at the end if not specified
    for epoch in range(1, cfg['epochs'] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, epoch, device)
        if scheduler is not None:
            scheduler.step()
        
        if epoch % eval_freq == 0:
            logger.info(f"--- Fast Evaluation at Epoch {epoch} ---")
            if cfg['dataset'] != "VinDrCXR":
                acc = evaluate(model, test_loader, device, alpha=cfg['alpha'])
                logger.info(f"Test Accuracy: {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    # Save best model logic can go here if needed.
            else:
                acc = evaluate_multilabel(model, test_loader, dataset.multi_map, device, alpha=cfg['alpha'])
                logger.info(f"Test Result: {acc}")

    # Save Config and OP automatically
    save_checkpoint(model, optimizer, scheduler, cfg, epoch, args.output_dir, is_best=True)

    if cfg['dataset'] != "VinDrCXR":
        acc = evaluate(model, test_loader, device, alpha=cfg['alpha'])

        logger.info(f"Hyparparams tuning...")
        val_acc, best_params = model.tune_hyperparameters(train_val_loader, device=device)
        logger.info(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")

        acc_tuned = evaluate(model, test_loader, device, alpha=best_params[0], t_end=best_params[1])
        logger.info(f"Test Accuracy (Before tuning): {acc:.2f}%")
        logger.info(f"Test Accuracy (After tuning): {acc_tuned:.2f}%")
    else:
        multi_map = dataset.multi_map
        acc = evaluate_multilabel(model, test_loader, multi_map, device, alpha=cfg['alpha'])

        logger.info(f"Hyparparams tuning...")
        val_acc, best_params = model.tune_hyperparameters(val_loader, multi_map=dataset.multi_map, device=device)
        logger.info(f"Best Hyparparams: alpha={best_params[0]}, timestep={best_params[1]}")

        acc_tuned = evaluate_multilabel(model, test_loader, multi_map, device, alpha=best_params[0], t_end=best_params[1])

        logger.info(f"Test Result (Before tuning): {acc}")
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