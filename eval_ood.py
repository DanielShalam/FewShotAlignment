"""OOD evaluation for FSA ImageNet models.

Loads a trained FSA checkpoint (trained on ImageNet with K-shot) and evaluates
on one of {ImageNetA, ImageNetR, ImageNetSketch, ImageNetV2}.

Key subtlety: ImageNet-A and ImageNet-R contain only 200 classes, so we build
a 200-way TextEncoder from the OOD dataset's classnames. The trained flow
adapter + OP were learned on 1000 classes, but they operate per-vector and
transfer directly (OP.W is [D_text, D_image], shape-invariant to class count).

Tuning protocol:
    - `--tune mode val_subset`: tune (alpha, t_end) on the subset of ImageNet
      val whose labels are in the OOD class set, re-labeled 0..C-1. This is
      the honest ID-protocol.
    - `--tune mode test`: tune on the OOD test itself (oracle upper bound).
    - `--tune mode none`: skip tuning, use fixed (alpha, t_end).

Example:
    python eval_ood.py \\
        --resume output/ImageNet/simple_lr5e5/model_best.pth \\
        --ood_dataset ImageNetA \\
        --output_dir output/ood_eval/A_seed42 \\
        --tune_mode val_subset \\
        config_overrides batch_size 128 text_adapter True flow_adapter simple \\
        seed 42 shots 16
"""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import yaml

from src.datasets.base_dataset import build_dataset, DatasetWrapper
from src.datasets.imagenet import ImageNet
from src.engine import evaluate
from src.model import FlowAdapter
from src.utils import load_checkpoint, set_seed, setup_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/imagenet_dinov3_qwen3.yaml")
    parser.add_argument("--resume", type=str, required=True, help="Path to trained model_best.pth")
    parser.add_argument("--ood_dataset", type=str, required=True,
                        )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tune_mode", type=str, default="val_subset",
                        choices=["val_subset", "test", "none"])
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Mixing coefficient when tune_mode=none (0 = pure flow, 1 = pure ZS)")
    parser.add_argument("--t_end", type=float, default=0.2,
                        help="Flow integration endpoint when tune_mode=none")
    parser.add_argument("--clip_zs_file", type=str, default=None,
                        help="Path to a .pth from compute_clip_zs_logits.py. If set, mix "
                             "the precomputed CLIP zero-shot logits into the test predictions.")
    parser.add_argument("--clip_zs_alpha", type=float, default=0.0,
                        help="Weight of CLIP ZS logits in the final mix (0 disables).")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Config overrides as key-value pairs")
    return parser.parse_args()


def load_cfg_with_overrides(config_path, opts):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if opts:
        if len(opts) % 2 != 0:
            raise ValueError("Override options must be key-value pairs")
        for i in range(0, len(opts), 2):
            k = opts[i].lstrip("-")
            v = opts[i+1]
            try:
                cfg[k] = yaml.safe_load(v)
            except Exception:
                cfg[k] = v
    return cfg


def build_ood_val_subset_loader(val_dataset_wrapper, ood_dataset, cfg, logger):
    """
    Return a DataLoader over the subset of ImageNet val whose labels are in
    ood_dataset's classes, remapped 0..C-1 to match the OOD text-feature order.
    """
    imagenet_root = os.path.join(
        os.path.abspath(os.path.expanduser(cfg["root"])), "imagenet")
    in1k_classnames = ImageNet.read_classnames(
        os.path.join(imagenet_root, "classnames.txt"))
    ood_classnames_in_order = list(ood_dataset.classnames)

    name_to_wnid = {v: k for k, v in in1k_classnames.items()}
    in1k_wnids = list(in1k_classnames.keys())
    in1k_wnid_to_idx = {w: i for i, w in enumerate(in1k_wnids)}
    ood_to_in1k = [in1k_wnid_to_idx[name_to_wnid[cn]] for cn in ood_classnames_in_order]
    in1k_to_ood = {in1k: ood for ood, in1k in enumerate(ood_to_in1k)}

    base = val_dataset_wrapper
    keep_indices = []
    for i in range(len(base)):
        item = base.data_source[i] if hasattr(base, "data_source") else None
        if item is None:
            break
        if int(item.label) in in1k_to_ood:
            keep_indices.append(i)
    logger.info(
        f"[OOD val subset] Kept {len(keep_indices)} / {len(base)} ImageNet val images "
        f"matching {len(in1k_to_ood)} OOD classes"
    )

    class _RelabelSubset(torch.utils.data.Dataset):
        def __init__(self, base, indices, remap):
            self.base = base
            self.indices = indices
            self.remap = remap
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            item = self.base[self.indices[i]]
            orig = int(item["label"]) if not torch.is_tensor(item["label"]) else int(item["label"])
            new_label = self.remap[orig]
            if torch.is_tensor(item["label"]):
                item["label"] = torch.tensor(new_label, dtype=item["label"].dtype)
            else:
                item["label"] = new_label
            return item

    subset = _RelabelSubset(base, keep_indices, in1k_to_ood)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 4),
        shuffle=False, drop_last=False, pin_memory=True,
    )


def main():
    args = get_args()
    cfg = load_cfg_with_overrides(args.config, args.opts)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(f"OOD eval on {args.ood_dataset} from checkpoint {args.resume}")
    logger.info(f"Tune mode: {args.tune_mode}")
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build OOD test dataset (classnames & test split)
    test_cfg = copy.deepcopy(cfg)
    test_cfg["dataset"] = args.ood_dataset
    ood_dataset = build_dataset(test_cfg)
    cfg["classnames"] = ood_dataset.classnames
    cfg["dataset"] = args.ood_dataset
    logger.info(f"OOD test: {len(ood_dataset.test)} images, {len(ood_dataset.classnames)} classes")

    # Build model (text encoder uses ood_dataset.classnames)
    model = FlowAdapter(cfg).to(device)
    load_checkpoint(model, args.resume)
    logger.info("Checkpoint loaded (adapter + t_adapter + OP.W)")

    # OOD test loader
    test_loader = torch.utils.data.DataLoader(
        DatasetWrapper(test_cfg, ood_dataset.test, transform=model.eval_tfm, is_train=False),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        drop_last=False, pin_memory=True,
    )

    # Optionally load external CLIP zero-shot logits for mixing
    ext_logits = None
    ext_alpha = float(args.clip_zs_alpha)
    if args.clip_zs_file is not None and ext_alpha > 0.0:
        zs_pkg = torch.load(args.clip_zs_file, map_location="cpu", weights_only=False)
        ext_logits_full = zs_pkg["logits"].float()  # [N_total, C]
        zs_ds = zs_pkg.get("dataset", "?")
        logger.info(f"Loaded CLIP ZS logits: {ext_logits_full.shape} from {args.clip_zs_file} "
                    f"(source dataset={zs_ds}, reported top-1={zs_pkg.get('top1_zs', '?')})")
        if ext_logits_full.size(0) != len(ood_dataset.test):
            raise ValueError(
                f"CLIP ZS logits length {ext_logits_full.size(0)} != test size {len(ood_dataset.test)}")
        ext_logits = ext_logits_full

    # Before-tuning eval
    acc_before = evaluate(
        model, test_loader, device, alpha=args.alpha, t_end=args.t_end,
        external_zs_logits=ext_logits, external_zs_alpha=ext_alpha,
    )
    tag = f"alpha={args.alpha}, t_end={args.t_end}"
    if ext_logits is not None:
        tag += f", clip_zs_alpha={ext_alpha}"
    logger.info(f"Test Accuracy ({tag}): {acc_before:.2f}%")

    out = {
        "ood_dataset": args.ood_dataset,
        "resume": args.resume,
        "tune_mode": args.tune_mode,
        "before_tune": {"alpha": args.alpha, "t_end": args.t_end, "acc": acc_before},
    }

    if args.tune_mode == "none":
        with open(Path(args.output_dir) / "results.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"DONE (no tuning). Test acc = {acc_before:.2f}%")
        return

    # Build val loader for tuning
    if args.tune_mode == "val_subset":
        # Need ImageNet val loader to subset
        imagenet_cfg = copy.deepcopy(cfg)
        imagenet_cfg["dataset"] = "ImageNet"
        imagenet_cfg["classnames"] = None  # not used here
        imagenet_dataset = build_dataset(imagenet_cfg)
        val_ds_wrapper = DatasetWrapper(
            imagenet_cfg, imagenet_dataset.val, transform=model.eval_tfm, is_train=False)
        tune_loader = build_ood_val_subset_loader(val_ds_wrapper, ood_dataset, cfg, logger)
    elif args.tune_mode == "test":
        logger.info("[DEBUG] Tuning on OOD test (oracle upper bound)")
        tune_loader = test_loader
    else:
        raise ValueError(args.tune_mode)

    # Tune
    val_acc, best_params = model.tune_hyperparameters(tune_loader, device=device)
    best_alpha, best_t_end, _ = best_params
    logger.info(f"Best (alpha, t_end) = ({best_alpha:.2f}, {best_t_end:.2f}); val_acc={val_acc*100:.2f}%")

    # Re-evaluate test with best (alpha, t_end)
    acc_after = evaluate(model, test_loader, device, alpha=best_alpha, t_end=best_t_end)
    logger.info(f"Test Accuracy (after tuning): {acc_after:.2f}%")

    out["after_tune"] = {
        "alpha": float(best_alpha), "t_end": float(best_t_end), "acc": float(acc_after),
    }
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"DONE. Before={acc_before:.2f}%  After={acc_after:.2f}%  (alpha={best_alpha}, t_end={best_t_end})")


if __name__ == "__main__":
    main()
