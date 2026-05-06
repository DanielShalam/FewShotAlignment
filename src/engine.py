import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()
def build_feature_cache(model, loader, device, desc="Caching feats"):
    """
    Run the frozen image encoder once over `loader` and return
    (img_feats, labels) tensors on `device`.

    img_feats are L2-normalized encoder outputs (post-normalize, pre-OP).
    This lets subsequent calls to model(cached_feats=...) skip the expensive
    image encoder forward entirely - useful for intermediate validation and
    hyperparameter tuning sweeps over (alpha, t_end).
    """
    model.eval()
    feats_list, labels_list = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch['img'].to(device, non_blocking=True)
        labels = batch['label'].to(device)
        f = F.normalize(model.image_encoder(images), dim=-1)
        feats_list.append(f)
        labels_list.append(labels)
    return torch.cat(feats_list, 0), torch.cat(labels_list, 0)


@torch.no_grad()
def evaluate_cached(model, cached_feats, cached_labels, device,
                    alpha=0., t_end=0.5, solver='midpoint', steps=2, chunk_size=512,
                    t_end_txt=None, tta_k=1, tta_noise_std=0.):
    """
    Evaluate top-1 accuracy using pre-computed image features.
    No image encoder forward; only flow ODE + matmul per chunk.

    If t_end_txt is None, keeps the default symmetric behaviour (text flows to 1 - t_end).

    Test-time feature augmentation (TTA):
      tta_k:          number of noisy replicas of each image feature. k=1 disables.
      tta_noise_std:  per-component Gaussian std added to x_0 before normalization.
    Logits are averaged across the k replicas.
    """
    import torch
    import torch.nn.functional as F
    model.eval()
    correct = 0
    total = 0
    do_tta = tta_k > 1 and tta_noise_std > 0.
    for i in range(0, cached_feats.size(0), chunk_size):
        f = cached_feats[i:i+chunk_size]
        y = cached_labels[i:i+chunk_size]
        B = f.size(0)

        if do_tta:
            f_rep = f.repeat_interleave(tta_k, dim=0)
            f_rep = F.normalize(f_rep + tta_noise_std * torch.randn_like(f_rep), dim=-1)
            out = model(cached_feats=f_rep, t_end=t_end, solver=solver, steps=steps,
                        t_end_txt=t_end_txt)
            logits_rep = (1 - alpha) * out['MT'] + alpha * out['ZS']
            logits = logits_rep.view(B, tta_k, -1).mean(dim=1)
        else:
            out = model(cached_feats=f, t_end=t_end, solver=solver, steps=steps,
                        t_end_txt=t_end_txt)
            logits = (1 - alpha) * out['MT'] + alpha * out['ZS']

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100. * correct / total




def train_one_epoch(model, loader, optimizer, epoch, device, multi_map=None, grad_clip=0.0):
    model.train()
    num_batches = len(loader)
    if num_batches == 0:
        raise ValueError(
            "Train loader has zero batches. This usually means dataset size is smaller than batch_size with drop_last=True. "
            "Reduce batch_size or disable drop_last for training."
        )

    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch['img'].to(device)

        if multi_map is not None:
            # Multi-label mode: build [B, C] target matrix from impaths.
            impaths = batch["impath"]
            labels = torch.stack([multi_map[p] for p in impaths], 0).to(device)
        else:
            labels = batch['label'].to(device)

        loss = model(images, labels=labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip), norm_type=2.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, device, alpha=0., t_end=0.5,  solver='midpoint', steps=2,
             external_zs_logits=None, external_zs_alpha=0.0):
    """Evaluate a trained FSA model.

    If ``external_zs_logits`` is provided, it should be a tensor of shape
    [N_total, C] where N_total equals the number of (non-shuffled) samples in
    ``loader``. It is interpreted as logits from an external zero-shot
    classifier (e.g. CLIP ViT-B/16 zero-shot) and mixed into the final
    prediction with weight ``external_zs_alpha``. The remaining weight is
    distributed (1 - alpha - external_zs_alpha) on MT and alpha on ZS, i.e.

        logits = (1 - alpha - external_zs_alpha) * MT
                 + alpha * ZS + external_zs_alpha * external_zs_logits

    The external logits are assumed to share the same scale (\~100 × cos-sim)
    as FSA's internal logits. Set external_zs_alpha=0 to recover the original
    behaviour.
    """
    model.eval()
    correct = 0
    total = 0
    use_ext = external_zs_logits is not None and float(external_zs_alpha) > 0.0
    if use_ext:
        external_zs_logits = external_zs_logits.to(device)
        # If model's logit_scale is 0 (i.e. exp=1, raw cosine logits), the MT
        # logits are at scale ~1 while external CLIP ZS logits are at scale ~100.
        # Rescale external logits to match MT's scale for fair mixing.
        if hasattr(model, 'logit_scale') and float(model.logit_scale.item()) == 0.0:
            external_zs_logits = external_zs_logits / 100.0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)

        out = model(images, t_end=t_end, solver=solver, steps=steps)  # Returns dict {'ZS': ..., 'MT': ...}

        # Ensemble logits
        B = labels.size(0)
        if use_ext:
            idx = torch.arange(total, total + B, device=device)
            ext = external_zs_logits[idx]
            logits = ((1 - alpha - external_zs_alpha) * out['MT']
                      + alpha * out['ZS']
                      + external_zs_alpha * ext)
        else:
            logits = (1 - alpha) * out['MT'] + alpha * out['ZS']

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


@torch.no_grad()
def evaluate_multi_alpha(model, loader, device, alphas,
                         t_end=0.5, solver='midpoint', steps=2,
                         external_zs_logits=None, mix_space='softmax',
                         base='MT', fsa_alphas=None):
    """Evaluate FSA across a list of CLIP-ZS mixing weights in one forward pass.

    ``base`` selects which FSA branch to mix with the external ZS logits:
        base='MT'  — mix flow-transported features logits (default)
        base='ZS'  — mix OP-only logits (no flow; uses out['ZS'])

    For each α in ``alphas``:
        if mix_space == 'softmax':
            p = (1 - α) * softmax(base) + α * softmax(external_zs_logits)
        else:  # 'logit'
            p = (1 - α) * base + α * external_zs_logits

    Returns a dict { α: accuracy_pct }.
    """
    assert external_zs_logits is not None, "external_zs_logits is required"
    assert base in ('MT', 'ZS'), f"base must be MT or ZS, got {base}"
    model.eval()
    external_zs_logits = external_zs_logits.to(device)

    # 2D mode: sweep over (fsa_alpha, clip_alpha)
    if fsa_alphas is not None:
        correct = {(fa, a): 0 for fa in fsa_alphas for a in alphas}
        total = 0
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch['img'].to(device)
            labels = batch['label'].to(device)
            B = labels.size(0)
            out = model(images, t_end=t_end, solver=solver, steps=steps)
            mt_logits = out['MT']
            zs_logits = out['ZS']
            idx = torch.arange(total, total + B, device=device)
            ext = external_zs_logits[idx]

            if mix_space == 'softmax':
                p_mt = F.softmax(mt_logits, dim=-1)
                p_zs = F.softmax(zs_logits, dim=-1)
                p_ext = F.softmax(ext, dim=-1)
                for fa in fsa_alphas:
                    p_fsa = (1 - fa) * p_mt + fa * p_zs  # internal FSA mix in softmax
                    for a in alphas:
                        p = (1 - a) * p_fsa + a * p_ext
                        pred = p.argmax(dim=-1)
                        correct[(fa, a)] += (pred == labels).sum().item()
            else:
                for fa in fsa_alphas:
                    fsa_logits = (1 - fa) * mt_logits + fa * zs_logits
                    for a in alphas:
                        logits = (1 - a) * fsa_logits + a * ext
                        pred = logits.argmax(dim=-1)
                        correct[(fa, a)] += (pred == labels).sum().item()
            total += B
        return {fa: {a: 100.0 * correct[(fa, a)] / total for a in alphas} for fa in fsa_alphas}

    # Original 1D mode: use `base` as the FSA-side input
    correct = {a: 0 for a in alphas}
    total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)
        B = labels.size(0)

        out = model(images, t_end=t_end, solver=solver, steps=steps)
        base_logits = out[base]  # [B, C]
        idx = torch.arange(total, total + B, device=device)
        ext = external_zs_logits[idx]  # [B, C]

        if mix_space == 'softmax':
            p_b = F.softmax(base_logits, dim=-1)
            p_ext = F.softmax(ext, dim=-1)
            for a in alphas:
                p = (1 - a) * p_b + a * p_ext
                pred = p.argmax(dim=-1)
                correct[a] += (pred == labels).sum().item()
        else:
            for a in alphas:
                logits = (1 - a) * base_logits + a * ext
                pred = logits.argmax(dim=-1)
                correct[a] += (pred == labels).sum().item()
        total += B

    return {a: 100.0 * correct[a] / total for a in alphas}


@torch.no_grad()
def evaluate_multilabel(model, loader, multi_map, device, alpha=0., t_end=0.8, solver="midpoint"):
    model.eval()

    scores, targets = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["img"].to(device)
        impaths = batch["impath"]
        y = torch.stack([multi_map[p] for p in impaths], 0).to(device)

        out = model(x, t_end=t_end, solver=solver)
        S = (1 - alpha) * out["MT"] + alpha * out["ZS"]
        scores.append(S.cpu().numpy())
        targets.append(y.cpu().numpy())

    S = np.concatenate(scores, 0)[:, :7]
    Y = np.concatenate(targets, 0)[:, :7]

    per_ap, per_roc = [], []
    for c in range(Y.shape[1]):
        if Y[:, c].sum() < 1:  # no positives
            per_ap.append(np.nan); per_roc.append(np.nan); continue
        per_ap.append(average_precision_score(Y[:, c], S[:, c]))
        try:
            per_roc.append(roc_auc_score(Y[:, c], S[:, c]))
        except ValueError:
            per_roc.append(np.nan)

    macro_ap  = float(np.nanmean(per_ap))
    macro_auc = float(np.nanmean(per_roc))
    return {
        "per_class_AP": per_ap,
        "macro_AUPRC": macro_ap,
        "macro_AUROC": macro_auc
    }

@torch.no_grad()
def evaluate_mlp(model, loader, device, alpha=0., **kwargs):
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)

        out = model(images)  # Returns dict {'ZS': ..., 'MT': ...}

        # Ensemble logits
        if model.OP.enable:
            logits = (1 - alpha) * out['MT'] + alpha * out['ZS']
        else:
            logits = out['MT']

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total

@torch.no_grad()
def evaluate_mlp_multilabel(model, loader, multi_map, device, alpha=0., **kwargs):
    model.eval()

    scores, targets = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["img"].to(device)
        impaths = batch["impath"]
        y = torch.stack([multi_map[p] for p in impaths], 0).to(device)

        out = model(x)
        
        if getattr(model, "OP", None) is not None and model.OP.enable:
            S = (1 - alpha) * out["MT"] + alpha * out["ZS"]
        else:
            S = out["MT"]

        scores.append(S.sigmoid().cpu().numpy())
        targets.append(y.cpu().numpy())

    S = np.concatenate(scores, 0)[:, :7]
    Y = np.concatenate(targets, 0)[:, :7]
    
    per_ap, per_roc = [], []
    for c in range(Y.shape[1]):
        if Y[:, c].sum() < 1:
            per_ap.append(np.nan); per_roc.append(np.nan); continue
        per_ap.append(average_precision_score(Y[:, c], S[:, c]))
        try:
            per_roc.append(roc_auc_score(Y[:, c], S[:, c]))
        except ValueError:
            per_roc.append(np.nan)

    macro_ap  = float(np.nanmean(per_ap))
    macro_auc = float(np.nanmean(per_roc))
    return {
        "per_class_AP": per_ap,
        "macro_AUPRC": macro_ap,
        "macro_AUROC": macro_auc
    }
