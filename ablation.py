"""
ablation.py — Batch runner for 2x2 ablation experiments and beta sweep

Experiment matrix:
  +-------------------+----------------------------+
  |                   |  T=8 (main)     T=1 (base) |
  +-------------------+----------------------------+
  | beta>0 (with VIB) |  * main         ablation-A |
  | beta=0 (no VIB)   |  ablation-B     ablation-C |
  +-------------------+----------------------------+

Beta sweep (T=8):
    beta_max in {0, 0.01, 0.1, 1.0}

Usage:
    python ablation.py                  # run everything
    python ablation.py --quick          # fast validation (500 steps)
    python ablation.py --only-main      # beta sweep only (T=8 configs)

Author: Puzhi YU
Date:   January 2026
"""

import os
import argparse
from dataclasses import replace

from train import TrainConfig, run_training
from monitor import plot_multi_run_comparison, plot_ignition_curves


# ──────────────────────────────────────────────
# 1. Experiment configurations
# ──────────────────────────────────────────────

def build_ablation_configs(total_steps: int = 2000) -> list:
    """Return all experiment configurations as (run_name, TrainConfig, label, color).
    Ordered: beta sweep first (main experiments), then T=1 ablation pair."""
    base = TrainConfig(total_steps=total_steps)

    configs = []

    # Beta sweep (T=8, main experiments)
    beta_colors = {
        0.00: '#95A5A6',    # gray: no bottleneck (control)
        0.01: '#AED6F1',    # light blue: weak compression
        0.10: '#2980B9',    # blue: medium compression (expected best)
        1.00: '#1A5276',    # dark blue: strong compression
    }
    for beta_max, color in beta_colors.items():
        run_name = f'T8_b{str(beta_max).replace(".", "")}'
        label    = f'T=8, β={beta_max}'
        cfg      = replace(base, T=8, beta_max=beta_max, run_name=run_name)
        configs.append((run_name, cfg, label, color))

    # Ablation A: T=1 + VIB (beta=0.1)
    cfg_a = replace(base, T=1, beta_max=0.1, run_name='T1_b01')
    configs.append(('T1_b01', cfg_a, 'T=1, β=0.1  (Ablation A)', '#E74C3C'))

    # Ablation C: T=1, no VIB (beta=0, pure baseline)
    cfg_c = replace(base, T=1, beta_max=0.0, run_name='T1_b0')
    configs.append(('T1_b0', cfg_c, 'T=1, β=0  (Ablation C / Baseline)', '#E8DAEF'))

    return configs


# ──────────────────────────────────────────────
# 2. Batch runner
# ──────────────────────────────────────────────

def run_ablation(
    total_steps: int  = 2000,
    only_main:   bool = False,
    checkpoint_dir: str = 'checkpoints',
    figures_dir:    str = 'figures',
):
    configs = build_ablation_configs(total_steps=total_steps)

    if only_main:
        # Beta sweep only (first 4 T=8 experiments)
        configs = [c for c in configs if 'T8' in c[0]]

    print("=" * 65)
    print(f"Ablation plan ({len(configs)} runs, {total_steps} steps each)")
    print("=" * 65)
    for run_name, cfg, label, _ in configs:
        print(f"  [{run_name}]  {label}")
    print("=" * 65)

    metrics_paths = {}   # run_name → metrics_path

    for run_name, cfg, label, color in configs:
        print(f"\n{'─'*65}")
        print(f"Starting: {label}")
        print(f"{'─'*65}")
        cfg_with_dir = replace(cfg, checkpoint_dir=checkpoint_dir)
        path = run_training(cfg_with_dir)
        metrics_paths[run_name] = path

        # Plot single-run figure immediately after completion
        plot_ignition_curves(path)

    # Comparative figures
    print(f"\n{'='*65}")
    print("Generating comparison figures...")
    os.makedirs(figures_dir, exist_ok=True)

    # Figure 1: beta sweep comparison (all T=8 runs, Nested sequences)
    beta_runs = [
        {'label': label, 'metrics_path': metrics_paths[run_name], 'color': color}
        for run_name, _, label, color in configs
        if 'T8' in run_name
    ]
    if len(beta_runs) > 1:
        plot_multi_run_comparison(
            run_configs=beta_runs,
            seq_type='nested',
            save_path=os.path.join(figures_dir, 'beta_sweep_nested.png'),
        )
        plot_multi_run_comparison(
            run_configs=beta_runs,
            seq_type='simple',
            save_path=os.path.join(figures_dir, 'beta_sweep_simple.png'),
        )

    # Figure 2: 2x2 ablation comparison (main vs. ablations A/B/C)
    ablation_runs = []
    key_runs = {
        'T8_b01':  ('★ Main: T=8, β=0.1',   '#2980B9'),
        'T8_b0':   ('Ablation B: T=8, β=0',  '#95A5A6'),
        'T1_b01':  ('Ablation A: T=1, β=0.1','#E74C3C'),
        'T1_b0':   ('Ablation C: T=1, β=0',  '#E8DAEF'),
    }
    for run_name, (label, color) in key_runs.items():
        if run_name in metrics_paths:
            ablation_runs.append({
                'label':        label,
                'metrics_path': metrics_paths[run_name],
                'color':        color,
            })
    if len(ablation_runs) > 1:
        plot_multi_run_comparison(
            run_configs=ablation_runs,
            seq_type='nested',
            save_path=os.path.join(figures_dir, 'ablation_2x2_nested.png'),
        )

    print(f"\n[OK] all ablation experiments complete. Figures saved to: {figures_dir}/")

    # Final accuracy summary
    print("\nFinal accuracy summary (last snapshot)")
    print("-" * 55)
    import json
    for run_name, path in metrics_paths.items():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        last = data['snapshots'][-1]
        print(f"  [{run_name:12s}]  "
              f"Simple={last['simple_acc']:.1%}  "
              f"Nested={last['nested_acc']:.1%}  "
              f"Random={last['random_acc']:.1%}")
    print("-" * 55)


# ──────────────────────────────────────────────
# 3. Command-line interface
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2x2 ablation experiment batch runner')
    parser.add_argument('--quick',       action='store_true',
                        help='Quick mode (500 steps, for pipeline validation)')
    parser.add_argument('--only-main',   action='store_true',
                        help='Run only the beta sweep (T=8 configs)')
    parser.add_argument('--total-steps', type=int, default=2000)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--figures-dir',    type=str, default='figures')
    args = parser.parse_args()

    steps = 500 if args.quick else args.total_steps

    run_ablation(
        total_steps=steps,
        only_main=args.only_main,
        checkpoint_dir=args.checkpoint_dir,
        figures_dir=args.figures_dir,
    )
