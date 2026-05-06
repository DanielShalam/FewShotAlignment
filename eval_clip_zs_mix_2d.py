"""Run a 2D (fsa_alpha, clip_alpha) softmax-mix sweep on a single checkpoint.

FSA internal mix: p_fsa = (1-fsa_alpha) * softmax(MT) + fsa_alpha * softmax(ZS)
Then: p = (1-clip_alpha) * p_fsa + clip_alpha * softmax(clip_zs_logits)
"""
import argparse, copy, json
from pathlib import Path
import torch, yaml

from src.datasets.base_dataset import build_dataset, DatasetWrapper
from src.engine import evaluate_multi_alpha
from src.model import FlowAdapter
from src.utils import load_checkpoint, set_seed


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/imagenet_dinov3_qwen3.yaml")
    p.add_argument("--resume", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--clip_zs_file", required=True)
    p.add_argument("--fsa_alphas", nargs="+", default=["0.0", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0"])
    p.add_argument("--clip_alphas", nargs="+", default=["0.0", "0.25", "0.5", "0.7", "0.9", "1.0"])
    p.add_argument("--mix_space", default="softmax", choices=["softmax", "logit"])
    p.add_argument("--t_end", type=float, default=0.2)
    p.add_argument("--output", required=True)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def load_cfg(path, opts):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if opts:
        for i in range(0, len(opts), 2):
            k = opts[i].lstrip("-"); v = opts[i+1]
            try: cfg[k] = yaml.safe_load(v)
            except: cfg[k] = v
    return cfg


def main():
    args = get_args()
    fsa_alphas = [float(a) for a in args.fsa_alphas]
    clip_alphas = [float(a) for a in args.clip_alphas]
    cfg = load_cfg(args.config, args.opts)
    cfg["dataset"] = args.dataset
    set_seed(cfg.get("seed", 42))
    device = "cuda"

    ds = build_dataset(copy.deepcopy(cfg))
    cfg["classnames"] = ds.classnames
    model = FlowAdapter(cfg).to(device)
    load_checkpoint(model, args.resume)

    loader = torch.utils.data.DataLoader(
        DatasetWrapper(cfg, ds.test, transform=model.eval_tfm, is_train=False),
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 8),
        drop_last=False, pin_memory=True,
    )
    zs = torch.load(args.clip_zs_file, map_location="cpu", weights_only=False)
    clip_logits = zs["logits"].float()

    result = evaluate_multi_alpha(
        model, loader, device, clip_alphas,
        t_end=args.t_end, external_zs_logits=clip_logits,
        mix_space=args.mix_space, fsa_alphas=fsa_alphas,
    )

    out = {"resume": args.resume, "dataset": args.dataset, "shots": cfg.get("shots"),
           "seed": cfg.get("seed"), "mix_space": args.mix_space,
           "fsa_alphas": fsa_alphas, "clip_alphas": clip_alphas,
           "accs": {str(fa): {str(ca): result[fa][ca] for ca in clip_alphas} for fa in fsa_alphas}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f: json.dump(out, f, indent=2)

    print("\n=== 2D sweep (row=fsa_alpha, col=clip_alpha) ===")
    label = "fsa\\clip"
    print(f'{label:>10} | ' + ' | '.join(f'{ca:>6.2f}' for ca in clip_alphas))
    for fa in fsa_alphas:
        row = ' | '.join(f'{result[fa][ca]:>6.2f}' for ca in clip_alphas)
        print(f'{fa:>10.2f} | {row}')
    # best point
    best = max(((fa, ca, result[fa][ca]) for fa in fsa_alphas for ca in clip_alphas), key=lambda x: x[2])
    print(f'\n[+] best: fsa_alpha={best[0]}, clip_alpha={best[1]}, acc={best[2]:.2f}')


if __name__ == "__main__":
    main()
