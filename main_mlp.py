import argparse
import os
from pathlib import Path
import copy
import yaml
import torch
from src.utils import setup_logger, set_seed, save_checkpoint, load_checkpoint

from src.datasets.base_dataset import build_dataset, build_loaders, DatasetWrapper
from src.model import ContrastiveMLPAdapter
from src.engine import train_one_epoch, evaluate_mlp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def main():
    args = get_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg["text_adapter"] = False  # Force no text adapter for MLP baseline
    cfg["lr"] = 0.0001

    if args.opts:
        if len(args.opts) % 2 != 0:
            raise ValueError("Override options must be key-value pairs (e.g., 'epochs 100 lr 0.001')")
        for i in range(0, len(args.opts), 2):
            k = args.opts[i].lstrip("-")
            v = args.opts[i+1]
            try:
                cfg[k] = yaml.safe_load(v)
            except Exception as e:
                print(e)
                cfg[k] = v

    if args.output_dir is None:
        cfg_name = args.config.split('/')[-1].split('.yaml')[0]
        args.output_dir = os.path.join("./output", cfg["dataset"], cfg_name + "_mlp", f"seed_{cfg['seed']}", f"shots_{cfg['shots']}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(f"Starting execution for seed {cfg['seed']} with config {args.config}")

    set_seed(cfg['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = ContrastiveMLPAdapter(cfg).to(device)

    logger.info(f"Building {cfg['dataset']} dataloaders...")

    train_loader, train_val_loader, val_loader, test_loader = build_loaders(
        train_cfg, dataset, model.train_tfm, model.eval_tfm, return_train_eval=True)

    if use_ood_eval:
        test_cfg = copy.deepcopy(train_cfg)
        test_cfg["dataset"] = ood_dataset_name
        ood_dataset = build_dataset(test_cfg)
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

    if args.resume:
        saved_cfg = load_checkpoint(model, args.resume)
    elif cfg["use_op"]:
        model.create_bank(train_val_loader)
        
    if args.eval_only:
        acc = evaluate_mlp(model, test_loader, device, alpha=cfg['alpha'])
        print(f"Test Accuracy (Before tuning): {acc:.2f}%")
        val_acc, best_params = model.tune_hyperparameters(val_loader, device=device)
        acc_tuned = evaluate_mlp(model, test_loader, device, alpha=best_params[0])
        print(f"Best Hyparparams: alpha={best_params[0]}")
        print(f"Test Accuracy (After tuning): {acc_tuned:.2f}%")
        return

    params = model.adapter.parameters() if not cfg["text_adapter"] \
        else list(model.adapter.parameters()) + list(model.t_adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    best_acc = 0.0
    eval_freq = cfg.get('eval_freq', cfg['epochs'])
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
            grad_clip=grad_clip,
        )
        if scheduler is not None:
            scheduler.step()
        
        if epoch % eval_freq == 0:
            print(f"--- Fast Evaluation at Epoch {epoch} ---")
            if cfg['dataset'] != "VinDrCXR":
                acc = evaluate_mlp(model, test_loader, device, alpha=cfg['alpha'])
                print(f"Test Accuracy: {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
            else:
                print("Multi-label evaluate skipped - not stubbed yet")

    save_checkpoint(model, optimizer, scheduler, cfg, epoch, args.output_dir, is_best=True)

    if cfg['dataset'] != "VinDrCXR":
        acc = evaluate_mlp(model, test_loader, device, alpha=cfg['alpha'])
        print(f"Test Accuracy (Before tuning): {acc:.2f}%")

        print("Hyparparams tuning...")
        val_acc, best_params = model.tune_hyperparameters(val_loader, device=device)
        print(f"Best Hyparparams: alpha={best_params[0]}")

        acc_tuned = evaluate_mlp(model, test_loader, device, alpha=best_params[0])
        print(f"Test Accuracy (After tuning): {acc_tuned:.2f}%")
    else:
        logger.info("Multi-label evaluation not fully stubbed for MLP baseline yet.")
        pass

    return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import logging
        logger = logging.getLogger("FlowAdapter")
        logger.exception("An error occurred during execution:")
        raise
