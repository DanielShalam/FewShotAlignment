import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.datasets.base_dataset import build_dataset, build_loaders
from src.model import FlowAdapter, MultiLabelFlowAdapter
from src.utils import load_checkpoint, setup_logger, set_seed


def get_args():
    parser = argparse.ArgumentParser(description="Ablate trained Flow models across ODE solvers and step counts")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, required=True, help="Path to checkpoint (checkpoint.pth/model_best.pth)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--solvers", nargs="+", default=["euler", "dopri5"])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--alpha", type=float, default=None, help="If unset, uses cfg['alpha']")
    parser.add_argument("--t_end", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None, help="Override cfg seed")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def evaluate_singlelabel(model, loader, device, alpha=0.0, t_end=0.5, solver="dopri5", steps=None):
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(loader, desc=f"Eval {solver} steps={steps}", leave=False):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        out_full = None
        for t_end in [0.25, 0.5, 0.75]:
            out = model(images, t_end=t_end, solver=solver, steps=steps)
            if out_full is None:
                out_full = out 
            else:
                for k in out_full.keys():
                    out_full[k] += out[k]
        logits = (1 - alpha) * out_full["MT"] + alpha * out_full["ZS"]

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


@torch.inference_mode()
def evaluate_multilabel(model, loader, multi_map, device, alpha=0.0, t_end=0.5, solver="dopri5", steps=None):
    model.eval()

    scores, targets = [], []
    for batch in tqdm(loader, desc=f"Eval {solver} steps={steps}", leave=False):
        x = batch["img"].to(device)
        impaths = batch["impath"]
        y = torch.stack([multi_map[p] for p in impaths], 0).to(device)

        out = model(x, t_end=t_end, solver=solver, steps=steps)
        s = (1 - alpha) * out["MT"] + alpha * out["ZS"]
        scores.append(s.cpu().numpy())
        targets.append(y.cpu().numpy())

    s = np.concatenate(scores, 0)
    y = np.concatenate(targets, 0)

    per_ap, per_roc = [], []
    for c in range(y.shape[1]):
        if y[:, c].sum() < 1:
            per_ap.append(np.nan)
            per_roc.append(np.nan)
            continue
        per_ap.append(average_precision_score(y[:, c], s[:, c]))
        try:
            per_roc.append(roc_auc_score(y[:, c], s[:, c]))
        except ValueError:
            per_roc.append(np.nan)

    return {
        "macro_AUPRC": float(np.nanmean(per_ap)),
        "macro_AUROC": float(np.nanmean(per_roc)),
    }


def main():
    args = get_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.opts:
        if len(args.opts) % 2 != 0:
            raise ValueError("Override options must be key-value pairs (e.g., 'epochs 100 lr 0.001')")
        for i in range(0, len(args.opts), 2):
            k = args.opts[i].lstrip("-")
            v = args.opts[i + 1]
            try:
                cfg[k] = yaml.safe_load(v)
            except Exception:
                cfg[k] = v

    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.output_dir is None:
        cfg_name = args.config.split("/")[-1].split(".yaml")[0]
        args.output_dir = os.path.join(
            "./output",
            cfg["dataset"],
            cfg_name,
            f"seed_{cfg['seed']}",
            f"shots_{cfg['shots']}",
            "solver_ablation",
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = build_dataset(cfg)
    cfg["classnames"] = dataset.classnames

    if cfg["dataset"] != "VinDrCXR":
        model = FlowAdapter(cfg).to(device)
    else:
        model = MultiLabelFlowAdapter(cfg).to(device)

    logger.info(f"Loading checkpoint from {args.resume}")
    load_checkpoint(model, args.resume)

    logger.info(f"Building {cfg['dataset']} dataloaders...")
    train_loader, train_val_loader, val_loader, test_loader = build_loaders(
        cfg, dataset, model.train_tfm, model.eval_tfm, return_train_eval=True
    )

    # Needed when inverse flow uses prototypes.
    if cfg.get("inv_fm_type", "none") == "proto":
        logger.info("inv_fm_type=proto -> creating support bank for prototypes")
        model.create_bank(train_val_loader)

    alpha = cfg["alpha"] if args.alpha is None else args.alpha

    rows = []
    for solver in args.solvers:
        solver_steps = [None] if solver == "dopri5" else [int(s) for s in args.steps]
        for steps in solver_steps:
            if cfg["dataset"] != "VinDrCXR":
                metric = evaluate_singlelabel(
                    model,
                    test_loader,
                    device=device,
                    alpha=alpha,
                    t_end=args.t_end,
                    solver=solver,
                    steps=steps,
                )
                row = {
                    "solver": solver,
                    "steps": steps,
                    "alpha": alpha,
                    "t_end": args.t_end,
                    "metric": "accuracy",
                    "value": metric,
                }
                logger.info(
                    f"solver={solver:>8} steps={str(steps):>5} alpha={alpha:.3f} t_end={args.t_end:.3f} -> acc={metric:.2f}%"
                )
            else:
                metric = evaluate_multilabel(
                    model,
                    test_loader,
                    multi_map=dataset.multi_map,
                    device=device,
                    alpha=alpha,
                    t_end=args.t_end,
                    solver=solver,
                    steps=steps,
                )
                row = {
                    "solver": solver,
                    "steps": steps,
                    "alpha": alpha,
                    "t_end": args.t_end,
                    "metric": "macro_AUPRC",
                    "value": metric["macro_AUPRC"],
                    "macro_AUROC": metric["macro_AUROC"],
                }
                logger.info(
                    f"solver={solver:>8} steps={str(steps):>5} alpha={alpha:.3f} t_end={args.t_end:.3f} "
                    f"-> macro_AUPRC={metric['macro_AUPRC']:.4f}, macro_AUROC={metric['macro_AUROC']:.4f}"
                )

            rows.append(row)

    csv_path = os.path.join(args.output_dir, "solver_ablation.csv")
    json_path = os.path.join(args.output_dir, "solver_ablation.json")

    if rows:
        keys = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)

    logger.info(f"Saved ablation table: {csv_path}")
    logger.info(f"Saved ablation json : {json_path}")


if __name__ == "__main__":
    main()
