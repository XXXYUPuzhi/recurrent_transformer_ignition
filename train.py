"""
train.py — Training loop with beta-annealing and periodic snapshots

Usage:
    # Single run
    python train.py --T 8 --beta-max 0.1 --run-name main_b01

    # Called programmatically by ablation.py
    from train import run_training
    run_training(config)

Saves to checkpoints/{run_name}/metrics.json every 100 steps:
    - Per-type accuracy, KL divergence, and RCR for Simple/Nested/Random
    - Current beta value and elapsed time

Author: Puzhi YU
Date:   January 2026
"""

import os
import json
import math
import argparse
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_generator import build_datasets, get_dataloaders, INT_TO_TYPE, N_VERTICES
from model import RecurrentTransformer


# ──────────────────────────────────────────────
# 1. Training configuration
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model hyperparameters
    d_model:        int   = 32
    nhead:          int   = 4
    bottleneck_dim: int   = 2
    T:              int   = 8       # recurrent steps (1 = no-recurrence baseline, 8 = main)
    L:              int   = 10      # sequence length

    # Training hyperparameters
    total_steps:    int   = 2000    # total training iterations
    batch_size:     int   = 128
    lr:             float = 1e-3
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0

    # VIB hyperparameters
    beta_max:       float = 0.1     # max beta (0 = no bottleneck pressure)
    warmup_ratio:   float = 0.5     # beta annealing warmup (first 50% for prediction, rest for compression)

    # Data
    n_per_type:     int   = 10_000
    seed:           int   = 42

    # Output
    run_name:       str   = 'run'
    snapshot_every: int   = 100     # evaluate every N steps
    checkpoint_dir: str   = 'checkpoints'


# ──────────────────────────────────────────────
# 2. Beta annealing schedule
# ──────────────────────────────────────────────

def get_beta(step: int, config: TrainConfig) -> float:
    """Beta annealing: beta(t) = beta_max * tanh(t / T_warmup).
    Starts at 0 and smoothly approaches beta_max."""
    if config.beta_max == 0.0:
        return 0.0
    warmup = max(1, int(config.warmup_ratio * config.total_steps))
    return config.beta_max * math.tanh(step / warmup)


# ──────────────────────────────────────────────
# 3. Single training step
# ──────────────────────────────────────────────

def train_step(
    model:     RecurrentTransformer,
    batch:     tuple,
    optimizer: torch.optim.Optimizer,
    beta:      float,
    device:    torch.device,
) -> Dict:
    """Forward + backward pass on a single batch; returns scalar metrics."""
    model.train()
    inputs, pred_labels, _ = batch
    inputs      = inputs.to(device)
    pred_labels = pred_labels.to(device)

    out    = model(inputs)
    logits = out['logits']         # [B, L, 8]

    B, L, C = logits.shape

    # Cross-entropy loss (independent next-vertex prediction at each position)
    ce_loss = F.cross_entropy(
        logits.reshape(B * L, C),
        pred_labels.reshape(B * L),
    )

    # VIB bottleneck penalty (mean KL across steps)
    kl_loss = torch.stack(out['kl_list']).mean()

    loss = ce_loss + beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(-1) == pred_labels).float().mean().item()

    return {
        'loss': loss.item(),
        'ce':   ce_loss.item(),
        'kl':   kl_loss.item(),
        'acc':  acc,
    }


# ──────────────────────────────────────────────
# 4. Snapshot evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_snapshot(
    model:            RecurrentTransformer,
    type_val_loaders: Dict,
    device:           torch.device,
) -> Dict:
    """Evaluate per-type (Simple/Nested/Random) Ignition metrics on validation set:
      acc : prediction accuracy
      kl  : mean VIB KL divergence
      rcr : recurrence convergence rate ||H_t - H_{t-1}|| (averaged over T steps)
    """
    model.eval()
    result = {}

    for type_name, loader in type_val_loaders.items():
        total_correct = total_tokens = 0
        kl_sum = rcr_sum = n_batches = 0

        for inputs, pred_labels, _ in loader:
            inputs      = inputs.to(device)
            pred_labels = pred_labels.to(device)

            out     = model(inputs)
            logits  = out['logits']
            B, L, C = logits.shape

            # Accuracy
            total_correct += (logits.argmax(-1) == pred_labels).sum().item()
            total_tokens  += B * L

            # KL (mean over T steps)
            kl_sum  += torch.stack(out['kl_list']).mean().item()
            # RCR (mean over T steps)
            rcr_sum += torch.stack(out['rcr_list']).mean().item()
            n_batches += 1

        result[type_name] = {
            'acc': total_correct / total_tokens,
            'kl':  kl_sum  / n_batches,
            'rcr': rcr_sum / n_batches,
        }

    return result


# ──────────────────────────────────────────────
# 5. Main training loop
# ──────────────────────────────────────────────

def run_training(config: TrainConfig) -> str:
    """Run the full training procedure.

    Returns:
        metrics_path: path to saved metrics.json (consumed by monitor.py)
    """
    # Output directory
    run_dir = os.path.join(config.checkpoint_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, 'metrics.json')

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{config.run_name}] device: {device}")

    # Data
    datasets, type_datasets = build_datasets(
        n_per_type=config.n_per_type,
        L=config.L,
        seed=config.seed,
    )
    train_loader = get_dataloaders(datasets, batch_size=config.batch_size)['train']

    # Per-type validation DataLoaders (for snapshot evaluation)
    type_val_loaders = {
        name: get_dataloaders(type_ds, batch_size=512)['val']
        for name, type_ds in type_datasets.items()
    }

    # Model
    torch.manual_seed(config.seed)
    model = RecurrentTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        bottleneck_dim=config.bottleneck_dim,
        T=config.T,
        L=config.L,
    ).to(device)
    print(f"[{config.run_name}] parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Logging structure
    log = {
        'config':    asdict(config),
        'snapshots': [],
    }

    # Training loop
    step      = 0
    best_acc  = 0.0
    t0        = time.time()
    train_iter = iter(train_loader)

    print(f"[{config.run_name}] starting training (total_steps={config.total_steps}, "
          f"beta_max={config.beta_max}, T={config.T})")
    print("-" * 65)

    while step < config.total_steps:
        # Reinitialize iterator at epoch boundary
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        beta = get_beta(step, config)
        metrics = train_step(model, batch, optimizer, beta, device)
        step += 1

        # Periodic snapshot evaluation
        if step % config.snapshot_every == 0 or step == config.total_steps:
            snap = evaluate_snapshot(model, type_val_loaders, device)
            elapsed = time.time() - t0

            # Detect Ignition (Simple or Nested accuracy > 50%)
            rule_acc = max(snap['simple']['acc'], snap['nested']['acc'])
            ignition_flag = rule_acc > 0.50

            log['snapshots'].append({
                'step':     step,
                'beta':     round(beta, 6),
                'elapsed':  round(elapsed, 1),
                **{f'{t}_{k}': round(v, 4)
                   for t, d in snap.items() for k, v in d.items()},
            })

            # Write metrics.json (live, monitor.py can read at any time)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(log, f, indent=2, ensure_ascii=False)

            # Save best checkpoint
            if rule_acc > best_acc:
                best_acc = rule_acc
                torch.save(
                    {'step': step, 'state_dict': model.state_dict(),
                     'config': asdict(config)},
                    os.path.join(run_dir, 'model_best.pt')
                )

            # Print progress
            ignition_str = " *** IGNITION ***" if ignition_flag else ""
            print(
                f"step {step:4d}/{config.total_steps} | "
                f"beta={beta:.4f} | "
                f"Simple={snap['simple']['acc']:.1%} | "
                f"Nested={snap['nested']['acc']:.1%} | "
                f"Random={snap['random']['acc']:.1%} | "
                f"KL={snap['simple']['kl']:.3f} | "
                f"RCR={snap['simple']['rcr']:.3f} | "
                f"{elapsed:.0f}s"
                f"{ignition_str}"
            )

    print("-" * 65)
    print(f"[{config.run_name}] training complete. Best accuracy: {best_acc:.1%}")
    print(f"[{config.run_name}] metrics saved: {metrics_path}")
    return metrics_path


# ──────────────────────────────────────────────
# 6. Command-line interface
# ──────────────────────────────────────────────

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description='Train RecurrentTransformer + VIB')
    parser.add_argument('--T',           type=int,   default=8)
    parser.add_argument('--beta-max',    type=float, default=0.1)
    parser.add_argument('--run-name',    type=str,   default='run')
    parser.add_argument('--total-steps', type=int,   default=2000)
    parser.add_argument('--batch-size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--n-per-type',  type=int,   default=10_000)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    return TrainConfig(
        T=args.T,
        beta_max=args.beta_max,
        run_name=args.run_name,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_per_type=args.n_per_type,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    config = parse_args()
    run_training(config)
