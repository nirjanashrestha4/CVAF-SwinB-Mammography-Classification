"""Ablation study runner for CVAF-SwinB.

Trains each ablation configuration in turn on the same data split and prints
a comparison table. Results are written to ablation_results.json in save_dir.
"""

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from model_ablation import build_ablation_model
from dataset import build_dataloaders


def run_epoch(model, loader, criterion, optimizer, scaler, device,
              grad_accum=1, is_train=True, scheduler=None, threshold=0.5):
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
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return total_loss / len(loader), bal_acc, auc, all_labels, all_preds, all_probs


def train_one_config(config_name, train_loader, val_loader, test_loader,
                     cfg, device):
    """Train one ablation configuration and return its test metrics."""
    print(f'\n{"=" * 65}')
    print(f'  Training: {config_name}')
    print(f'{"=" * 65}')

    gc.collect()
    torch.cuda.empty_cache()

    common = dict(num_classes=1, img_size=cfg.img_size,
                  pretrained=True, head_dropout=0.3)
    if config_name != 'SingleView':
        common.update(dict(num_heads=8, dropout=0.1))

    model = build_ablation_model(config_name, **common).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Params: {n_params:.1f}M')

    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

    save_dir = Path(cfg.save_dir) / config_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    patience_cnt = 0
    patience = getattr(cfg, 'patience', 15)

    print(f"  {'Epoch':>5} | {'Tr AUC':>7} | {'Val AUC':>8} | {'Time':>6}")
    print(f"  {'-' * 35}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        _, _, tr_auc, *_ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accum=cfg.grad_accum, is_train=True, scheduler=scheduler,
        )
        _, _, val_auc, *_ = run_epoch(
            model, val_loader, criterion, None, None, device, is_train=False,
        )
        elapsed = time.time() - t0
        print(f'  {epoch:>5} | {tr_auc:>7.4f} | {val_auc:>8.4f} | {elapsed:>5.1f}s')

        raw = model.module if hasattr(model, 'module') else model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_cnt = 0
            torch.save({'epoch': epoch, 'state_dict': raw.state_dict(),
                        'val_auc': val_auc}, save_dir / 'best_model.pt')
            print(f'         New best val AUC = {best_val_auc:.4f}')
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f'  Early stopping at epoch {epoch}.')
                break

    print(f'\n  Loading best checkpoint (val AUC={best_val_auc:.4f}) ...')
    ckpt = torch.load(save_dir / 'best_model.pt',
                      map_location=device, weights_only=False)
    raw = model.module if hasattr(model, 'module') else model
    raw.load_state_dict(ckpt['state_dict'])

    _, _, test_auc, test_labels, test_preds, _ = run_epoch(
        model, test_loader, criterion, None, None, device, is_train=False)

    sensitivity = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    specificity = recall_score(test_labels, test_preds, pos_label=0, zero_division=0)
    f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
    precision = precision_score(test_labels, test_preds, pos_label=1, zero_division=0)
    bal_acc = balanced_accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    result = {
        'config': config_name,
        'best_epoch': int(ckpt['epoch']),
        'val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1': float(f1),
        'precision': float(precision),
        'balanced_acc': float(bal_acc),
        'n_params_M': round(n_params, 1),
        'cm': cm.tolist(),
    }

    print(f'\n  TEST: AUC={test_auc:.4f}  Sens={sensitivity:.4f}  '
          f'Spec={specificity:.4f}  F1={f1:.4f}')
    return result


def run_ablation(cfg):
    """Train all configurations in cfg.configs on a shared data split."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Configs: {cfg.configs}')
    print(f'Epochs : {cfg.epochs}  |  patience: {cfg.patience}')
    print(f'pos_weight: {cfg.pos_weight}')

    print('\nBuilding dataloaders ...')
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
    print(f'Train: {len(train_loader.dataset):,}  '
          f'Val: {len(val_loader.dataset):,}  '
          f'Test: {len(test_loader.dataset):,}')

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for config_name in cfg.configs:
        result = train_one_config(
            config_name, train_loader, val_loader, test_loader, cfg, device)
        all_results[config_name] = result

        with open(save_dir / 'ablation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print('  Saved ablation_results.json')

    sep = '=' * 80
    print(f'\n\n{sep}')
    print('  ABLATION STUDY — COMPARISON TABLE')
    print(sep)
    print(f"  {'Config':<16} {'AUC':>7} {'Sens':>7} {'Spec':>7} "
          f"{'F1':>7} {'BalAcc':>8} {'Params':>8}")
    print(f"  {'-' * 65}")
    for name, r in all_results.items():
        marker = ' <- proposed' if name == 'FullCVAF' else ''
        print(f"  {name:<16} {r['test_auc']:>7.4f} {r['sensitivity']:>7.4f} "
              f"{r['specificity']:>7.4f} {r['f1']:>7.4f} "
              f"{r['balanced_acc']:>8.4f} {r['n_params_M']:>7.1f}M{marker}")
    print(sep)

    return all_results