"""Re-score finished DeepLoc2 grid runs with F1-max (TorchDrug/VenusFactory convention).

Replays test-set inference using a reloaded FlowAdapter checkpoint,
then computes F1-max on (score, target) at the (best alpha, best tau) from the run log.
Prints macro-AUPRC (sklearn) and F1-max per run for direct comparison with VenusFactory.
"""
import argparse, json, os, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

def f1_max(pred: torch.Tensor, target: torch.Tensor) -> float:
    # TorchDrug/VenusFactory implementation
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order); inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten(); recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return float(all_f1.max())

def macro_auprc(Y: np.ndarray, S: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    aps = [average_precision_score(Y[:, c], S[:, c]) for c in range(Y.shape[1]) if Y[:, c].sum() > 0]
    return float(np.mean(aps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/deeploc")
    ap.add_argument("--grid", required=True, help="directory containing run subdirs + .log files")
    ap.add_argument("--config_dir", default="configs")
    args = ap.parse_args()

    # We re-build the model per run, load the saved adapter state, run test inference,
    # and accumulate (score, target) across the test loader at the tuned (alpha, t_end).
    import yaml
    from src.datasets.base_dataset import build_dataset, build_loaders, FEATURE_REGISTRY
    from src.model import MultiLabelFlowAdapter
    from src.utils import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = Path(args.grid)

    rows = []
    for log in sorted(grid.glob("*.log")):
        tag = log.stem
        # parse tag: esm150_{text}_k{K}_seed{seed}
        m = re.match(r"esm150_(\w+)_k(\d+)_seed(\d+)$", tag)
        if not m: continue
        txt, K, seed = m.group(1), int(m.group(2)), int(m.group(3))

        txt_path = {
            "pubmed": "data/deeploc/class_text_biomednlp_pubmedbert_base_uncased_abstract_fulltext.pt",
            "qwen06": "data/deeploc/class_text_qwen3_embedding_0_6b.pt",
            "qwen8b": "data/deeploc/class_text_qwen3_embedding_8b.pt",
        }[txt]
        img_dim = 640

        best = re.findall(r"Best Hyparparams: alpha=([0-9.]+), timestep=([0-9.]+)", log.read_text())
        if not best:
            continue
        alpha, t_end = float(best[-1][0]), float(best[-1][1])

        # Build a cfg mimicking the YAML used
        cfg = yaml.safe_load(open(f"{args.config_dir}/deeploc2_esm150_qwen.yaml"))
        cfg.update(dict(
            dataset="DeepLoc2", root=str(Path(args.data_dir)),
            esm_file="seq_embed_esm2_t30_150m_ur50d.pt",
            text_features_path=txt_path, img_dim=img_dim,
            shots=K, seed=seed, batch_size=64, num_workers=0,
        ))
        FEATURE_REGISTRY.clear()
        dataset = build_dataset(cfg)
        cfg["classnames"] = dataset.classnames
        _, _, _, test_loader = build_loaders(cfg, dataset, None, None, return_train_eval=True)

        model = MultiLabelFlowAdapter(cfg).to(device)
        ckpt_dir = grid / tag
        ckpts = list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.pt"))
        if not ckpts:
            print(f"SKIP {tag}: no checkpoint"); continue
        load_checkpoint(model, str(ckpts[0]))
        model.eval()

        all_S, all_Y = [], []
        with torch.inference_mode():
            for batch in test_loader:
                imgs = batch["img"].to(device)
                impaths = batch["impath"]
                y = torch.stack([dataset.multi_map[p] for p in impaths], 0)
                logits = model(imgs, t_end=t_end)
                S = (1 - alpha) * logits["MT"] + alpha * logits["ZS"]
                all_S.append(S.cpu()); all_Y.append(y)
        S = torch.cat(all_S, 0); Y = torch.cat(all_Y, 0)
        mAP = macro_auprc(Y.numpy(), S.numpy())
        fm = f1_max(S, Y)
        rows.append((txt, K, seed, alpha, t_end, mAP, fm))
        print(f"{tag:40s}  α={alpha:.2f} τ={t_end:.2f}  mAUPRC={mAP*100:6.2f}  F1max={fm*100:6.2f}")

    # aggregate
    print()
    agg = defaultdict(list)
    for (txt, K, seed, a, t, mAP, fm) in rows:
        agg[(txt, K)].append((mAP, fm))
    print(f"{'txt':8s} {'K':>3s}  {'mAUPRC':>13s}  {'F1max':>13s}")
    for (txt, K), vs in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        m = np.array([v[0] for v in vs])*100; f = np.array([v[1] for v in vs])*100
        print(f"{txt:8s} {K:3d}  {m.mean():6.2f}±{m.std():4.2f}  {f.mean():6.2f}±{f.std():4.2f}")

if __name__ == "__main__":
    main()
