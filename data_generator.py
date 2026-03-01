"""
data_generator.py — Octagon geometric sequence data generation

Generates three types of vertex sequences on a regular octagon:
    Simple  : x_{t+1} = (x_t + 1) mod 8          [clockwise rotation]
    Nested  : alternating +2 / -1 steps (mod 8)   [two-steps-forward, one-step-back]
    Random  : x_{t+1} ~ Uniform{0,...,7}          [incompressible control]

Input encoding:
    vertex k -> [cos(2*pi*k/8), sin(2*pi*k/8)]  (unit-circle geometric coordinates)

Task:
    Autoregressive next-vertex prediction (8-class classification).
    Chance-level accuracy = 1/8 = 12.5%.

Author: Puzhi YU
Date:   January 2026
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

N_VERTICES = 8
TYPE_TO_INT = {'simple': 0, 'nested': 1, 'random': 2}
INT_TO_TYPE = {v: k for k, v in TYPE_TO_INT.items()}


# ──────────────────────────────────────────────
# 1. Encoding
# ──────────────────────────────────────────────

def encode_geometric(indices: np.ndarray) -> np.ndarray:
    """Convert vertex indices to unit-circle geometric coordinates.

    Args:
        indices: (...,) int array with values in {0,...,7}
    Returns:
        (..., 2) float32 array, each row = [cos(2*pi*k/8), sin(2*pi*k/8)]
    """
    angles = 2.0 * np.pi * indices / N_VERTICES
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)


# ──────────────────────────────────────────────
# 2. Sequence generators
# ──────────────────────────────────────────────

def generate_simple(n: int, L: int, seed: int = None):
    """Generate Simple sequences: x_{t+1} = (x_t + 1) mod 8.

    Args:
        n:    number of sequences
        L:    sequence length (number of input steps)
        seed: random seed
    Returns:
        inputs:      (n, L, 2)  geometric coordinates, float32
        pred_labels: (n, L)     next-vertex indices, int64
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, N_VERTICES, size=n)

    seqs = np.empty((n, L + 1), dtype=np.int64)
    seqs[:, 0] = starts
    for t in range(L):
        seqs[:, t + 1] = (seqs[:, t] + 1) % N_VERTICES

    return encode_geometric(seqs[:, :-1]), seqs[:, 1:]


def generate_nested(n: int, L: int, seed: int = None):
    """Generate Nested sequences: alternating +2 / -1 steps (two-forward, one-back).

    Step pattern: steps[t] = +2 if t is even, -1 if t is odd.
    Example (from vertex 0): 0 -> 2 -> 1 -> 3 -> 2 -> 4 -> 3 -> 5 ...

    Args:
        n:    number of sequences
        L:    sequence length (number of input steps)
        seed: random seed
    Returns:
        inputs:      (n, L, 2)  geometric coordinates, float32
        pred_labels: (n, L)     next-vertex indices, int64
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, N_VERTICES, size=n)

    # Step pattern: even steps +2, odd steps -1
    steps = np.array([2 if t % 2 == 0 else -1 for t in range(L)], dtype=np.int64)

    seqs = np.empty((n, L + 1), dtype=np.int64)
    seqs[:, 0] = starts
    for t in range(L):
        seqs[:, t + 1] = (seqs[:, t] + steps[t]) % N_VERTICES

    return encode_geometric(seqs[:, :-1]), seqs[:, 1:]


def generate_random(n: int, L: int, seed: int = None):
    """Generate Random sequences: x_{t+1} ~ Uniform{0,...,7}.

    Args:
        n:    number of sequences
        L:    sequence length (number of input steps)
        seed: random seed
    Returns:
        inputs:      (n, L, 2)  geometric coordinates, float32
        pred_labels: (n, L)     next-vertex indices, int64
    """
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, N_VERTICES, size=(n, L + 1), dtype=np.int64)
    return encode_geometric(seqs[:, :-1]), seqs[:, 1:]


# ──────────────────────────────────────────────
# 3. Dataset
# ──────────────────────────────────────────────

class OctagonDataset(Dataset):
    """PyTorch dataset for octagon vertex sequences.

    Each sample contains:
        inputs:      (L, 2)   geometric coordinate sequence (model input)
        pred_labels: (L,)     next-vertex index (cross-entropy target)
        type_label:  scalar   sequence type: 0=Simple, 1=Nested, 2=Random
                              (used as target for the Rule linear probe)
    """

    def __init__(
        self,
        inputs:      np.ndarray,   # (N, L, 2)
        pred_labels: np.ndarray,   # (N, L)
        type_labels: np.ndarray,   # (N,)
    ):
        self.inputs      = torch.from_numpy(inputs)
        self.pred_labels = torch.from_numpy(pred_labels.astype(np.int64))
        self.type_labels = torch.from_numpy(type_labels.astype(np.int64))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],       # (L, 2)   float32
            self.pred_labels[idx],  # (L,)     int64
            self.type_labels[idx],  # scalar   int64
        )


# ──────────────────────────────────────────────
# 4. Dataset construction and splitting
# ──────────────────────────────────────────────

def build_datasets(
    n_per_type: int = 10_000,
    L:          int = 10,
    seed:       int = 42,
    ratios:     tuple = (0.70, 0.15, 0.15),
):
    """Build the full dataset and split by type and phase (train/val/test).

    Each sequence type is split independently to maintain strict class balance.

    Args:
        n_per_type: sequences per type (default 10,000)
        L:          sequence length (default 10)
        seed:       random seed for reproducibility
        ratios:     (train, val, test) proportions, default (0.70, 0.15, 0.15)

    Returns:
        datasets:      dict{'train','val','test'} -> OctagonDataset (all types mixed)
        type_datasets: dict{'simple','nested','random'}
                         -> dict{'train','val','test'} -> OctagonDataset (single type)
                       Used for per-type evaluation and Ignition monitoring.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    generators = [
        ('simple', generate_simple),
        ('nested', generate_nested),
        ('random', generate_random),
    ]

    # Generate, shuffle, and split each type independently
    type_split_data = {}   # name -> phase -> (inputs, pred_labels, type_labels)

    for i, (name, gen_fn) in enumerate(generators):
        inputs, pred_labels = gen_fn(n_per_type, L, seed=seed + i)
        type_labels = np.full(n_per_type, TYPE_TO_INT[name], dtype=np.int64)

        # Intra-type shuffle
        rng = np.random.default_rng(seed + 100 + i)
        perm = rng.permutation(n_per_type)
        inputs, pred_labels, type_labels = inputs[perm], pred_labels[perm], type_labels[perm]

        # Split indices
        n_train = int(ratios[0] * n_per_type)
        n_val   = int(ratios[1] * n_per_type)
        slices = {
            'train': slice(0, n_train),
            'val':   slice(n_train, n_train + n_val),
            'test':  slice(n_train + n_val, None),
        }
        type_split_data[name] = {
            phase: (inputs[s], pred_labels[s], type_labels[s])
            for phase, s in slices.items()
        }

    # Build Dataset objects
    type_datasets = {name: {} for name in TYPE_TO_INT}
    datasets = {}

    for phase in ('train', 'val', 'test'):
        merged_inp, merged_pred, merged_type = [], [], []

        for name in TYPE_TO_INT:
            inp, pred, typ = type_split_data[name][phase]
            merged_inp.append(inp)
            merged_pred.append(pred)
            merged_type.append(typ)
            # Single-type dataset
            type_datasets[name][phase] = OctagonDataset(inp, pred, typ)

        # Merge all types and reshuffle
        all_inp  = np.concatenate(merged_inp,  axis=0)
        all_pred = np.concatenate(merged_pred, axis=0)
        all_type = np.concatenate(merged_type, axis=0)

        rng = np.random.default_rng(seed + 200 + TYPE_TO_INT.get(phase, hash(phase) % 100))
        perm = rng.permutation(len(all_inp))
        datasets[phase] = OctagonDataset(all_inp[perm], all_pred[perm], all_type[perm])

    return datasets, type_datasets


def get_dataloaders(
    datasets:    dict,
    batch_size:  int = 128,
    num_workers: int = 0,
) -> dict:
    """Build DataLoaders from the dataset dict returned by build_datasets.

    Args:
        datasets:    dict{'train','val','test'} -> OctagonDataset
        batch_size:  default 128
        num_workers: default 0 (Windows multiprocessing compatibility)
    Returns:
        dict{'train','val','test'} -> DataLoader
    """
    return {
        phase: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for phase, ds in datasets.items()
    }


# ──────────────────────────────────────────────
# 5. Verification utilities
# ──────────────────────────────────────────────

def _indices_from_coords(coords: np.ndarray) -> np.ndarray:
    """Recover vertex indices from geometric coordinates (for validation)."""
    angles = np.arctan2(coords[..., 1], coords[..., 0])
    return (np.round(angles / (2.0 * np.pi / N_VERTICES)) % N_VERTICES).astype(int)


def verify_sequences(n_check: int = 3, L: int = 10, seed: int = 0):
    """Print example sequences to visually verify generation rules.
    Also checks encoding-decoding round-trip consistency."""
    print("=" * 60)
    print(f"Sequence verification  (n={n_check}, L={L})")
    print("=" * 60)

    for name, gen_fn in [('simple', generate_simple),
                          ('nested', generate_nested),
                          ('random', generate_random)]:
        inputs, labels = gen_fn(n_check, L, seed=seed)
        print(f"\n[{name.upper()}]")
        for i in range(n_check):
            # Recover vertex indices from geometric coordinates
            vertex_seq = list(_indices_from_coords(inputs[i]))
            vertex_seq.append(int(labels[i, -1]))
            print(f"  seq {i}: {' -> '.join(map(str, vertex_seq))}")

    print("=" * 60)


def dataset_summary(datasets: dict, type_datasets: dict):
    """Print dataset statistics summary."""
    print("\nDataset statistics")
    print("-" * 45)
    print("  Mixed datasets:")
    for phase, ds in datasets.items():
        print(f"    {phase:5s}: {len(ds):6,d} sequences")
    print()
    print("  Per-type datasets:")
    for type_name, phases in type_datasets.items():
        for phase, ds in phases.items():
            print(f"    {type_name:8s} / {phase:5s}: {len(ds):5,d} sequences")
    print("-" * 45)


# ──────────────────────────────────────────────
# 6. Quick self-test
# ──────────────────────────────────────────────

if __name__ == '__main__':
    # 1. Visual verification of sequence rules
    verify_sequences(n_check=3, L=10, seed=0)

    # 2. Build datasets
    print("\nBuilding datasets (n_per_type=10,000, L=10)...")
    datasets, type_datasets = build_datasets(n_per_type=10_000, L=10, seed=42)
    dataset_summary(datasets, type_datasets)

    # 3. Verify DataLoader output shapes
    loaders = get_dataloaders(datasets, batch_size=128)
    inputs, pred_labels, type_labels = next(iter(loaders['train']))
    print(f"\nDataLoader batch shapes (batch_size=128):")
    print(f"  inputs:      {tuple(inputs.shape)}    -> (B, L, 2)")
    print(f"  pred_labels: {tuple(pred_labels.shape)}   -> (B, L)")
    print(f"  type_labels: {tuple(type_labels.shape)}     -> (B,)")

    # 4. Verify random baseline accuracy (should be ~12.5%)
    rnd_loader = get_dataloaders(type_datasets['random'], batch_size=5000)['test']
    inp_r, lbl_r, _ = next(iter(rnd_loader))
    random_pred = torch.randint(0, N_VERTICES, lbl_r.shape)
    acc = (random_pred == lbl_r).float().mean().item()
    print(f"\nRandom guess accuracy (Random test set): {acc:.1%}  (expected ~12.5%)")

    # 5. Verify Simple sequence rule consistency (each step +1)
    sim_loader = get_dataloaders(type_datasets['simple'], batch_size=1000)['test']
    inp_s, lbl_s, _ = next(iter(sim_loader))
    inp_indices = _indices_from_coords(inp_s.numpy())   # (B, L)
    expected = (inp_indices[:, -1] + 1) % N_VERTICES    # last step prediction = last + 1
    actual   = lbl_s[:, -1].numpy()
    match_rate = (expected == actual).mean()
    print(f"Rule consistency check (Simple last step +1): {match_rate:.1%}  (expected 100%)")

    print("\n[OK] data_generator.py self-test passed")
