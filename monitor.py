"""
monitor.py — Visualization of Ignition dynamics from training metrics

Reads metrics.json snapshots and plots the three Ignition indicators
(Accuracy, KL divergence, RCR) as synchronized time-series curves.

Usage:
    python monitor.py --metrics checkpoints/main_b01/metrics.json

    # Programmatic interface (used by ablation.py)
    from monitor import plot_ignition_curves, plot_multi_run_comparison

Outputs:
    ignition_curves.png   — 3-panel figure for a single run
    comparison.png        — multi-run overlay (called from ablation.py)

Author: Puzhi YU
Date:   January 2026
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless backend, works on servers without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional

# Color scheme (consistent across all figures)
COLORS = {
    'simple': '#E74C3C',    # red
    'nested': '#3498DB',    # blue
    'random': '#95A5A6',    # gray
}
METRIC_COLORS = {
    'acc': '#E74C3C',
    'kl':  '#3498DB',
    'rcr': '#2ECC71',
}


# ──────────────────────────────────────────────
# 1. Data loading utilities
# ──────────────────────────────────────────────

def load_metrics(metrics_path: str) -> Dict:
    """Load metrics.json, returns config and snapshots."""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_series(snapshots: List[Dict], seq_type: str, metric: str) -> tuple:
    """Extract (steps, values) time series for a given type and metric."""
    steps  = [s['step'] for s in snapshots]
    values = [s[f'{seq_type}_{metric}'] for s in snapshots]
    return np.array(steps), np.array(values)


def detect_ignition_step(steps: np.ndarray, acc: np.ndarray, threshold: float = 0.5) -> Optional[int]:
    """Find the first training step where accuracy exceeds the threshold (ignition point)."""
    mask = acc > threshold
    if mask.any():
        return int(steps[np.argmax(mask)])
    return None


# ──────────────────────────────────────────────
# 2. Main figure: Ignition curves for a single run
# ──────────────────────────────────────────────

def plot_ignition_curves(
    metrics_path: str,
    save_path:    Optional[str] = None,
    show:         bool = False,
) -> str:
    """Plot three synchronized Ignition indicators during training.

    Three panels:
      Panel 1: Prediction accuracy (Simple / Nested / Random)
      Panel 2: Mean VIB KL divergence
      Panel 3: Recurrence Convergence Rate (RCR)

    Returns:
        save_path: path to saved figure
    """
    data      = load_metrics(metrics_path)
    config    = data['config']
    snapshots = data['snapshots']

    if save_path is None:
        run_dir   = os.path.dirname(metrics_path)
        save_path = os.path.join(run_dir, 'ignition_curves.png')

    fig = plt.figure(figsize=(12, 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)
    ax_acc = fig.add_subplot(gs[0])
    ax_kl  = fig.add_subplot(gs[1])
    ax_rcr = fig.add_subplot(gs[2])

    ignition_step = None   # earliest ignition step (accuracy first > 50%)

    for seq_type in ['simple', 'nested', 'random']:
        color = COLORS[seq_type]
        ls    = '--' if seq_type == 'random' else '-'
        lw    = 1.5  if seq_type == 'random' else 2.0

        steps_acc, acc = extract_series(snapshots, seq_type, 'acc')
        steps_kl,  kl  = extract_series(snapshots, seq_type, 'kl')
        steps_rcr, rcr = extract_series(snapshots, seq_type, 'rcr')

        label = seq_type.capitalize()
        ax_acc.plot(steps_acc, acc * 100, color=color, ls=ls, lw=lw, label=label)
        ax_kl.plot(steps_kl,  kl,         color=color, ls=ls, lw=lw, label=label)
        ax_rcr.plot(steps_rcr, rcr,        color=color, ls=ls, lw=lw, label=label)

        # Detect ignition step (for Simple / Nested sequences)
        if seq_type in ('simple', 'nested'):
            ig = detect_ignition_step(steps_acc, acc)
            if ig is not None:
                if ignition_step is None or ig < ignition_step:
                    ignition_step = ig

    # Mark ignition point with vertical line
    if ignition_step is not None:
        for ax in [ax_acc, ax_kl, ax_rcr]:
            ax.axvline(ignition_step, color='gold', lw=2.0, ls=':', alpha=0.9,
                       label=f'Ignition (~step {ignition_step})')

    # Random baseline reference line
    ax_acc.axhline(12.5, color='black', lw=1.0, ls=':', alpha=0.4, label='Random baseline (12.5%)')

    # Axis configuration
    T        = config['T']
    beta_max = config['beta_max']
    title    = f"Ignition Curves — T={T}, β_max={beta_max}  ({config['run_name']})"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    ax_acc.set_ylabel('Accuracy (%)', fontsize=11)
    ax_acc.set_ylim(-5, 105)
    ax_acc.set_yticks([0, 12.5, 25, 50, 75, 100])
    ax_acc.legend(fontsize=9, ncol=4, loc='upper left')
    ax_acc.grid(alpha=0.3)

    ax_kl.set_ylabel('Mean KL Divergence', fontsize=11)
    ax_kl.legend(fontsize=9, ncol=4, loc='upper right')
    ax_kl.grid(alpha=0.3)

    ax_rcr.set_ylabel('RCR  ‖H_t − H_{t−1}‖', fontsize=11)
    ax_rcr.set_xlabel('Training Steps', fontsize=11)
    ax_rcr.legend(fontsize=9, ncol=4, loc='upper right')
    ax_rcr.grid(alpha=0.3)

    for ax in [ax_acc, ax_kl, ax_rcr]:
        ax.set_xlim(left=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[monitor] figure saved: {save_path}")
    if show:
        plt.show()
    plt.close()
    return save_path


# ──────────────────────────────────────────────
# 3. Comparison figure: multi-run overlay (used by ablation.py)
# ──────────────────────────────────────────────

def plot_multi_run_comparison(
    run_configs:  List[Dict],   # [{'label': str, 'metrics_path': str, 'color': str}, ...]
    seq_type:     str = 'nested',
    save_path:    str = 'figures/comparison.png',
    show:         bool = False,
) -> str:
    """Overlay Accuracy / KL / RCR from multiple runs on the same figure.

    Args:
        run_configs: list of dicts, each specifying label, metrics_path, color
        seq_type:    which sequence type to plot (default 'nested', shows clearest contrast)
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Ablation Comparison  ({seq_type.capitalize()} sequences)',
                 fontsize=13, fontweight='bold')

    metrics_names = ['acc', 'kl', 'rcr']
    ylabels       = ['Accuracy (%)', 'Mean KL Divergence', 'RCR ‖H_t − H_{t−1}‖']

    for run in run_configs:
        data      = load_metrics(run['metrics_path'])
        snapshots = data['snapshots']
        color     = run.get('color', None)
        label     = run['label']

        for ax, metric, ylabel in zip(axes, metrics_names, ylabels):
            steps, vals = extract_series(snapshots, seq_type, metric)
            if metric == 'acc':
                vals = vals * 100
            ax.plot(steps, vals, label=label, color=color, lw=2.0)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel('Training Steps', fontsize=10)
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Random baseline
    axes[0].axhline(12.5, color='black', lw=1.0, ls=':', alpha=0.4)
    axes[0].set_ylim(-5, 105)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center',
               ncol=len(run_configs), fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[monitor] comparison figure saved: {save_path}")
    if show:
        plt.show()
    plt.close()
    return save_path


# ──────────────────────────────────────────────
# 4. Command-line interface
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Ignition monitoring curves')
    parser.add_argument('--metrics', type=str, required=True,
                        help='Path to metrics.json')
    parser.add_argument('--save',    type=str, default=None,
                        help='Output image path (default: same dir as metrics.json)')
    parser.add_argument('--show',    action='store_true',
                        help='Open interactive plot window (requires GUI)')
    args = parser.parse_args()

    plot_ignition_curves(args.metrics, save_path=args.save, show=args.show)
