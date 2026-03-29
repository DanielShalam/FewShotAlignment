import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, epoch, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch['img'].to(device)
        labels = batch['label'].to(device)

        loss = model(images, labels=labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, device, alpha=0., t_end=0.5):
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)

        out = model(images, t_end=t_end, solver='dopri5')  # Returns dict {'ZS': ..., 'MT': ...}

        # Ensemble logits
        logits = (1 - alpha) * out['MT'] + alpha * out['ZS']

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


@torch.inference_mode()
def evaluate_multilabel(model, loader, multi_map, device, alpha=0., t_end=0.5, solver="dopri5"):
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

    S = np.concatenate(scores, 0)
    Y = np.concatenate(targets, 0)

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

@torch.inference_mode()
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
