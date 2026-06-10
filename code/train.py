"""Training script for CVAF-SwinB.

Trains the model with mixup, label smoothing and a warmup-cosine learning
rate schedule. After training it reloads the best checkpoint, selects a
decision threshold on the validation set, and reports test metrics at both
the default and selected thresholds.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from model import CVAFSwinB
from dataset import build_dataloaders


def mixup_batch(views, labels, alpha=0.2, device='cuda'):
    """Blend pairs of samples within a batch."""
    if alpha <= 0:
        return views, labels
    lam = np.random.beta(alpha, alpha)
    batch_size = labels.size(0)
    idx = torch.randperm(batch_size, device=device)
    mixed_views = {k: lam * v + (1 - lam) * v[idx] for k, v in views.items()}
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_views, mixed_labels


def run_epoch(
    model, loader, criterion, optimizer, scaler, device,
    grad_accum=1, is_train=True, scheduler=None,
    threshold=0.5, use_mixup=False,
):
    """Run one epoch and return loss, balanced accuracy, AUC and arrays."""
    model.train(is_train)
    total_loss = 0.0
    all_probs: list = []
    all_labels: list = []

    if is_train:
        optimizer.zero_grad()

    for step, (views, labels) in enumerate(loader):
        views = {k: v.to(device, non_blocking=True) for k, v in views.items()}
        labels = labels.to(device, non_blocking=True)

        if is_train and use_mixup:
            views, labels = mixup_batch(views, labels, alpha=0.2, device=device)

        with autocast('cuda', enabled=(scaler is not None)):
            logits = model(lcc=views['lcc'], lmlo=views['lmlo'],
                           rcc=views['rcc'], rmlo=views['rmlo'])
            loss = criterion(logits, labels.float().unsqueeze(1))
            if is_train:
                loss = loss / grad_accum

        if is_train:
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        probs = torch.sigmoid(logits.detach()).squeeze(1)
        total_loss += loss.item() * (grad_accum if is_train else 1)
        all_probs.extend(probs.cpu().float().numpy())
        all_labels.extend(labels.cpu().float().numpy().round().astype(int))

    avg_loss = total_loss / len(loader)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)
    all_preds = (all_probs >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return avg_loss, bal_acc, auc, all_labels, all_preds, all_probs


def find_best_threshold(labels, probs, metric='f1'):
    """Search thresholds in [0.05, 0.95) and return the best for a metric."""
    best_thresh = 0.5
    best_score = 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(labels, preds, pos_label=1, zero_division=0)
        else:
            score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return float(best_thresh), float(best_score)


def evaluate(labels, preds, probs, title='Test Set Results', threshold=0.5):
    """Print and return a full set of metrics for one threshold."""
    auc = roc_auc_score(labels, probs)
    bal_acc = balanced_accuracy_score(labels, preds)
    sensitivity = recall_score(labels, preds, pos_label=1, zero_division=0)
    specificity = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(labels, preds)

    sep = '=' * 60
    print(f'\n{sep}')
    print(f'  {title}')
    print(f'  Decision threshold: {threshold:.3f}')
    print(sep)
    print(f'  AUC-ROC              : {auc:.4f}')
    print(f'  Balanced Accuracy    : {bal_acc:.4f}')
    print(f'  Sensitivity (Recall) : {sensitivity:.4f}')
    print(f'  Specificity          : {specificity:.4f}')
    print(f'  Precision            : {precision:.4f}')
    print(f'  F1 (Malignant)       : {f1:.4f}')
    print(f'\n{classification_report(labels, preds, target_names=["Benign", "Malignant"], digits=4)}')
    print('  Confusion Matrix:')
    print(f'    TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}')
    print(f'    FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}')
    print(sep)

    return {
        'auc': float(auc),
        'balanced_accuracy': float(bal_acc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1': float(f1),
        'precision': float(precision),
        'threshold': float(threshold),
        'confusion_matrix': cm.tolist(),
    }


class SmoothBCELoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing and positive-class weighting."""

    def __init__(self, pos_weight, smoothing=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.smoothing = smoothing

    def forward(self, logits, targets):
        smooth = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return self.bce(logits, smooth)


def train(cfg) -> float:
    """Train CVAF-SwinB and return the test AUC at the selected threshold."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = getattr(cfg, 'patience', 10)

    print(f'Device     : {device}')
    print(f'Epochs     : {cfg.epochs}  |  patience={patience}')
    print(f'pos_weight : {cfg.pos_weight}\n')

    print('Building dataloaders ...')
    train_loader, val_loader, test_loader = build_dataloaders(
        cbis_root=getattr(cfg, 'cbis_root', None),
        vindr_root=cfg.vindr_root,
        vindr_images=getattr(cfg, 'vindr_images', None),
        vindr_labels=getattr(cfg, 'vindr_labels', None),
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    print(f'  Train: {len(train_loader.dataset):,}  '
          f'Val: {len(val_loader.dataset):,}  '
          f'Test: {len(test_loader.dataset):,}\n')

    print('Loading CVAF-SwinB ...')
    model = CVAFSwinB(
        num_classes=1, img_size=cfg.img_size, pretrained=True,
        num_heads=8, dropout=0.1, head_dropout=0.3,
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs')

    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model params: {n:.1f}M\n')

    label_smoothing = getattr(cfg, 'label_smoothing', 0.05)
    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    criterion = SmoothBCELoss(pos_weight, label_smoothing)

    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
    other_params = [p for nm, p in model.named_parameters()
                    if 'backbone' not in nm and p.requires_grad]
    optimizer = AdamW([
        {'params': backbone.parameters(), 'lr': cfg.lr * 0.1},
        {'params': other_params, 'lr': cfg.lr},
    ], weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * (len(train_loader) // cfg.grad_accum)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    patience_cnt = 0
    history = {k: [] for k in ['train_loss', 'train_auc', 'train_bal_acc',
                               'val_loss', 'val_auc', 'val_bal_acc']}

    header = (f"{'Epoch':>6} | {'Tr Loss':>8} {'Tr AUC':>8} {'Tr BAcc':>8} | "
              f"{'Val Loss':>9} {'Val AUC':>8} {'Val BAcc':>8} | "
              f"{'LR':>10} | {'Time':>6}")
    dash = '-' * len(header)
    print(f'Scheduler: {warmup_steps} warmup / {total_steps} total steps\n')
    print(dash)
    print(header)
    print(dash)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_auc, *_ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accum=cfg.grad_accum, is_train=True,
            scheduler=scheduler, use_mixup=True,
        )
        val_loss, val_acc, val_auc, val_labels, val_preds, val_probs = run_epoch(
            model, val_loader, criterion, None, None, device, is_train=False,
        )

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[1]['lr']

        for k, v in zip(
            ['train_loss', 'train_auc', 'train_bal_acc',
             'val_loss', 'val_auc', 'val_bal_acc'],
            [tr_loss, tr_auc, tr_acc, val_loss, val_auc, val_acc],
        ):
            history[k].append(v)

        print(f'{epoch:>6} | {tr_loss:>8.4f} {tr_auc:>8.4f} {tr_acc:>8.4f} | '
              f'{val_loss:>9.4f} {val_auc:>8.4f} {val_acc:>8.4f} | '
              f'{lr_now:>10.2e} | {elapsed:>5.1f}s')

        raw = model.module if hasattr(model, 'module') else model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_cnt = 0
            torch.save({'epoch': epoch, 'state_dict': raw.state_dict(),
                        'val_auc': val_auc, 'cfg': vars(cfg),
                        'val_labels': val_labels, 'val_probs': val_probs},
                       save_dir / 'best_model.pt')
            print(f'         New best val AUC = {best_val_auc:.4f}  (saved)')
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f'  Early stopping at epoch {epoch} '
                      f'(patience={patience}).')
                print(f'  Best val AUC = {best_val_auc:.4f}')
                break

        torch.save({'epoch': epoch, 'state_dict': raw.state_dict(),
                    'optimizer': optimizer.state_dict(), 'history': history,
                    'cfg': vars(cfg)}, save_dir / 'last_model.pt')

    print(dash)
    print(f'\nTraining complete. Best val AUC = {best_val_auc:.4f}')

    pd.DataFrame(history).to_csv(
        save_dir / 'training_history.csv', index_label='epoch')

    print('\nLoading best checkpoint ...')
    ckpt = torch.load(save_dir / 'best_model.pt',
                      map_location=device, weights_only=False)
    raw = model.module if hasattr(model, 'module') else model
    raw.load_state_dict(ckpt['state_dict'])

    val_labels_saved = ckpt.get('val_labels')
    val_probs_saved = ckpt.get('val_probs')

    if val_labels_saved is not None:
        opt_thresh, opt_f1 = find_best_threshold(
            val_labels_saved, val_probs_saved, metric='f1')
        print(f'Selected threshold (val F1={opt_f1:.4f}): {opt_thresh:.3f}')
    else:
        opt_thresh = 0.5
        print('Using default threshold 0.5')

    _, _, _, test_labels, _, test_probs = run_epoch(
        model, test_loader, criterion, None, None, device, is_train=False)

    test_preds_default = (test_probs >= 0.5).astype(int)
    results_default = evaluate(
        test_labels, test_preds_default, test_probs,
        title=f'Test Results — epoch {ckpt["epoch"]} — threshold=0.50',
        threshold=0.5,
    )

    test_preds_opt = (test_probs >= opt_thresh).astype(int)
    results_opt = evaluate(
        test_labels, test_preds_opt, test_probs,
        title=f'Test Results — epoch {ckpt["epoch"]} — threshold={opt_thresh:.3f}',
        threshold=opt_thresh,
    )

    all_results = {
        'best_epoch': ckpt['epoch'],
        'val_auc': float(ckpt['val_auc']),
        'default_threshold': results_default,
        'selected_threshold': results_opt,
        'selected_thresh_value': float(opt_thresh),
    }
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    return results_opt.get('auc', 0.0)