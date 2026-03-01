"""
analysis.py — Post-training mechanistic analysis

Provides three analysis tools for probing what the model has learned:

  1. Linear Probing
     - Probe P (Position): decode current vertex index from H_t (8-class)
     - Probe R (Rule):     decode sequence rule type from H_t (3-class)
     - Track probe accuracy across recurrent steps t=0..T

  2. Generalization Test
     - Zero-shot evaluation on unseen rules (+3, -2, +4, -3 step sizes)

  3. Bottleneck 2D Visualization
     - Scatter plot of z_final in R^2, colored by vertex index
     - Reveals whether the bottleneck recapitulates octagon geometry

Usage:
    python analysis.py --checkpoint checkpoints/main_b01/model_best.pt

Author: Puzhi YU
Date:   January 2026
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_generator import (
    build_datasets, get_dataloaders,
    generate_simple, generate_nested, generate_random,
    encode_geometric, OctagonDataset, N_VERTICES, TYPE_TO_INT,
)
from model import RecurrentTransformer


# ──────────────────────────────────────────────
# Utility: load model from checkpoint
# ──────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> RecurrentTransformer:
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']
    model  = RecurrentTransformer(
        d_model=config['d_model'],
        nhead=config['nhead'],
        bottleneck_dim=config['bottleneck_dim'],
        T=config['T'],
        L=config['L'],
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"[analysis] model loaded: step={ckpt['step']}, T={config['T']}, "
          f"beta_max={config['beta_max']}")
    return model, config


# ──────────────────────────────────────────────
# 1. Linear probing
# ──────────────────────────────────────────────

@torch.no_grad()
def collect_hidden_states(
    model:   RecurrentTransformer,
    loader:  DataLoader,
    device:  torch.device,
) -> dict:
    """Collect hidden states at each recurrent step t=0..T and z_final.

    Returns:
        {
          'h_steps':     list of T+1 arrays, each [N_total, L, d_model]
          'z_final':     array [N_total, L, bottleneck_dim]
          'pos_labels':  array [N_total, L]   current vertex index (for probe P)
          'type_labels': array [N_total]      sequence type (for probe R)
        }
    """
    T       = model.T
    h_steps = [[] for _ in range(T + 1)]
    z_list  = []
    pos_list  = []   # current vertex indices (from input, for probe P)
    type_list = []

    for inputs, pred_labels, type_labels in loader:
        inputs = inputs.to(device)
        out    = model(inputs)

        # Hidden states (T+1 per sample, including initial embedding H_0)
        for t, h in enumerate(out['h_list']):
            h_steps[t].append(h.numpy())    # [B, L, d_model]

        z_list.append(out['z_final'].numpy())   # [B, L, bottleneck_dim]

        # Position labels (recovered from input coordinates)
        angles = torch.atan2(inputs[..., 1], inputs[..., 0]).cpu().numpy()
        pos    = (np.round(angles / (2.0 * np.pi / N_VERTICES)) % N_VERTICES).astype(int)
        pos_list.append(pos)                     # [B, L]
        type_list.append(type_labels.numpy())    # [B]

    return {
        'h_steps':     [np.concatenate(h, axis=0) for h in h_steps],  # T+1 × [N, L, d]
        'z_final':     np.concatenate(z_list,  axis=0),                # [N, L, 2]
        'pos_labels':  np.concatenate(pos_list, axis=0),               # [N, L]
        'type_labels': np.concatenate(type_list, axis=0),              # [N]
    }


def run_linear_probe(
    model:         RecurrentTransformer,
    type_datasets: dict,
    device:        torch.device,
    save_dir:      str = 'figures',
) -> dict:
    """Train linear probes at each recurrent step t=0..T:
      - Probe P (Position): predict current vertex position (8-class)
      - Probe R (Rule):     predict sequence rule type (3-class)

    Returns:
        {'P': [T+1 accuracies], 'R': [T+1 accuracies]}
    """
    T = model.T
    print(f"\n[analysis] linear probing (T={T} steps + initial embedding = {T+1} states)")

    # Use mixed data from all three types (ensures probe R has balanced classes)
    # Training set for fitting, validation set for evaluation
    all_loaders_train = {
        name: DataLoader(ds['train'], batch_size=512, shuffle=False)
        for name, ds in type_datasets.items()
    }
    all_loaders_val = {
        name: DataLoader(ds['val'], batch_size=512, shuffle=False)
        for name, ds in type_datasets.items()
    }

    # Collect hidden states from train and val sets
    def collect_all(loaders):
        collected = None
        for name, loader in loaders.items():
            c = collect_hidden_states(model, loader, device)
            if collected is None:
                collected = c
            else:
                collected['h_steps'] = [
                    np.concatenate([collected['h_steps'][t], c['h_steps'][t]], axis=0)
                    for t in range(T + 1)
                ]
                collected['z_final']     = np.concatenate([collected['z_final'],     c['z_final']],     axis=0)
                collected['pos_labels']  = np.concatenate([collected['pos_labels'],  c['pos_labels']],  axis=0)
                collected['type_labels'] = np.concatenate([collected['type_labels'], c['type_labels']], axis=0)
        return collected

    train_data = collect_all(all_loaders_train)
    val_data   = collect_all(all_loaders_val)

    probe_acc = {'P': [], 'R': []}

    for t in range(T + 1):
        # Flatten [N, L, d_model] -> [N*L, d_model]
        N, L, d = train_data['h_steps'][t].shape

        X_train_flat = train_data['h_steps'][t].reshape(N * L, d)
        X_val_flat   = val_data['h_steps'][t].reshape(val_data['h_steps'][t].shape[0] * L, d)

        # Probe P: position labels (flattened to match)
        y_P_train = train_data['pos_labels'].reshape(-1)
        y_P_val   = val_data['pos_labels'].reshape(-1)

        # Probe R: rule labels (same label for all L positions in a sequence)
        N_train = train_data['type_labels'].shape[0]
        N_val   = val_data['type_labels'].shape[0]
        y_R_train = np.repeat(train_data['type_labels'], L)  # [N*L]
        y_R_val   = np.repeat(val_data['type_labels'],   L)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_flat)
        X_val_s   = scaler.transform(X_val_flat)

        # Probe P (position)
        clf_P = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs',
                                   multi_class='multinomial', random_state=0)
        clf_P.fit(X_train_s, y_P_train)
        acc_P = clf_P.score(X_val_s, y_P_val)

        # Probe R (rule type)
        clf_R = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs',
                                   multi_class='multinomial', random_state=0)
        clf_R.fit(X_train_s, y_R_train)
        acc_R = clf_R.score(X_val_s, y_R_val)

        probe_acc['P'].append(acc_P)
        probe_acc['R'].append(acc_R)
        print(f"  Step t={t:2d}: Probe-P(Position)={acc_P:.1%}  Probe-R(Rule)={acc_R:.1%}")

    # Plot probe accuracy vs. recurrent step
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x_ticks = list(range(T + 1))
    ax.plot(x_ticks, [a * 100 for a in probe_acc['P']], 'o-',
            color='#E74C3C', lw=2, label='Probe P: Position (8-class)')
    ax.plot(x_ticks, [a * 100 for a in probe_acc['R']], 's-',
            color='#3498DB', lw=2, label='Probe R: Rule (3-class)')
    ax.axhline(100 / 8,  color='#E74C3C', ls=':', alpha=0.5, lw=1)   # chance level P: 12.5%
    ax.axhline(100 / 3,  color='#3498DB', ls=':', alpha=0.5, lw=1)   # chance level R: 33.3%
    ax.set_xlabel('Recurrent Step t', fontsize=11)
    ax.set_ylabel('Linear Probe Accuracy (%)', fontsize=11)
    ax.set_title('Linear Probing at Each Recurrent Step\n'
                 '(Position probe ↓ + Rule probe ↑ = Information Transfer)', fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xticks(x_ticks)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path = os.path.join(save_dir, 'linear_probe.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[analysis] linear probe figure saved: {save_path}")
    return probe_acc


# ──────────────────────────────────────────────
# 2. Generalization test
# ──────────────────────────────────────────────

def generate_new_rule(rule_step: int, n: int, L: int, seed: int = 99):
    """Generate sequences with a novel rule (unseen during training):
    x_{t+1} = (x_t + rule_step) mod 8."""
    rng    = np.random.default_rng(seed)
    starts = rng.integers(0, N_VERTICES, size=n)
    seqs   = np.empty((n, L + 1), dtype=np.int64)
    seqs[:, 0] = starts
    for t in range(L):
        seqs[:, t + 1] = (seqs[:, t] + rule_step) % N_VERTICES
    inputs = encode_geometric(seqs[:, :-1])
    labels = seqs[:, 1:]
    return inputs, labels


@torch.no_grad()
def run_generalization_test(
    model:  RecurrentTransformer,
    device: torch.device,
    L:      int = 10,
    n:      int = 2000,
) -> dict:
    """Test zero-shot generalization on rules unseen during training.

    Returns:
        dict mapping rule_name -> accuracy
    """
    model.eval()
    test_rules = {
        '+3 steps':          3,
        '-2 steps':         -2,
        '+4 steps (mirror)': 4,
        '-3 steps':         -3,
    }

    print("\n[analysis] generalization test")
    results = {}

    for rule_name, step_size in test_rules.items():
        inputs_np, labels_np = generate_new_rule(step_size, n, L, seed=99)
        inputs_t = torch.from_numpy(inputs_np).to(device)
        labels_t = torch.from_numpy(labels_np.astype(np.int64))

        out  = model(inputs_t)
        logits = out['logits']   # [B, L, 8]
        acc  = (logits.argmax(-1).cpu() == labels_t).float().mean().item()
        results[rule_name] = acc
        print(f"  {rule_name:25s}: {acc:.1%}")

    return results


# ──────────────────────────────────────────────
# 3. Bottleneck 2D visualization
# ──────────────────────────────────────────────

@torch.no_grad()
def plot_bottleneck_2d(
    model:         RecurrentTransformer,
    type_datasets: dict,
    device:        torch.device,
    save_dir:      str = 'figures',
    n_samples:     int = 300,
) -> str:
    """Scatter plot of bottleneck vectors z_final in R^2, colored by vertex index.
    After ignition, the 8 vertices should form a clear octagonal geometry."""
    model.eval()
    print("\n[analysis] bottleneck 2D visualization")

    # Use Simple sequences only (clearest structure)
    ds     = type_datasets['simple']['test']
    loader = DataLoader(ds, batch_size=n_samples, shuffle=True)
    inputs, pred_labels, _ = next(iter(loader))
    inputs = inputs.to(device)

    out     = model(inputs)
    z       = out['z_final'].numpy()       # [B, L, 2]
    # Recover input position indices from coordinates
    angles  = np.arctan2(inputs[..., 1].cpu().numpy(),
                         inputs[..., 0].cpu().numpy())
    pos_idx = (np.round(angles / (2.0 * np.pi / N_VERTICES)) % N_VERTICES).astype(int)

    # Flatten: one point per time-step position
    z_flat   = z.reshape(-1, 2)           # [B*L, 2]
    pos_flat = pos_idx.reshape(-1)        # [B*L]

    cmap   = cm.get_cmap('tab10', N_VERTICES)
    colors = [cmap(p) for p in pos_flat]

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(z_flat[:, 0], z_flat[:, 1],
                         c=pos_flat, cmap='tab10', vmin=0, vmax=N_VERTICES - 1,
                         alpha=0.5, s=15, edgecolors='none')

    # Annotate true vertex positions on the unit circle
    for k in range(N_VERTICES):
        angle = 2 * np.pi * k / N_VERTICES
        ax.annotate(str(k), xy=(np.cos(angle) * 0.05, np.sin(angle) * 0.05),
                    fontsize=14, ha='center', va='center',
                    color=cmap(k), fontweight='bold')

    plt.colorbar(scatter, ax=ax, ticks=range(N_VERTICES), label='Vertex Index')
    ax.set_title('Bottleneck Representation z_final ∈ ℝ²\n'
                 '(After Ignition: vertices should form octagon geometry)',
                 fontsize=11)
    ax.set_xlabel('z[0]', fontsize=11)
    ax.set_ylabel('z[1]', fontsize=11)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'bottleneck_2d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[analysis] bottleneck visualization saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────
# 4. Run all analyses
# ──────────────────────────────────────────────

def run_all_analysis(
    checkpoint_path: str,
    save_dir:        str = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(checkpoint_path, device)

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(checkpoint_path), 'figures')

    # Rebuild dataset with the same seed for consistency
    _, type_datasets = build_datasets(
        n_per_type=config['n_per_type'],
        L=config['L'],
        seed=config['seed'],
    )

    # 1. Linear probes
    probe_acc = run_linear_probe(model, type_datasets, device, save_dir=save_dir)

    # 2. Generalization test
    gen_results = run_generalization_test(model, device, L=config['L'])

    # 3. Bottleneck 2D visualization
    plot_bottleneck_2d(model, type_datasets, device, save_dir=save_dir)

    # Summary
    print("\n" + "=" * 55)
    print("Analysis Summary")
    print("=" * 55)
    print(f"Linear probes (final step t={model.T}):")
    print(f"  Probe P (Position): {probe_acc['P'][-1]:.1%}")
    print(f"  Probe R (Rule):     {probe_acc['R'][-1]:.1%}")
    print("\nGeneralization test (novel rules):")
    for rule, acc in gen_results.items():
        print(f"  {rule:25s}: {acc:.1%}")
    print("=" * 55)
    print(f"[OK] all analyses complete, figures saved to: {save_dir}")


# ──────────────────────────────────────────────
# 5. Command-line interface
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-training mechanistic analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model_best.pt')
    parser.add_argument('--save-dir',   type=str, default=None,
                        help='Output directory for figures (default: figures/ next to checkpoint)')
    args = parser.parse_args()

    run_all_analysis(args.checkpoint, save_dir=args.save_dir)
