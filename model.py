"""
model.py — RecurrentTransformer with Variational Information Bottleneck

Implements a weight-sharing recurrent Transformer (single layer, T shared-weight
iterations) coupled with a readout-stage VIB layer. The architecture is designed
as a minimal computational model for studying "Ignition" dynamics in the spirit
of Dehaene's Global Neuronal Workspace theory.

Architecture:
    Input [B, L=10, 2]
      -> Linear embedding (2 -> d_model=32)
      -> Sinusoidal positional encoding
      -> T recurrent steps (shared TransformerEncoderLayer + causal mask)
      -> VIB readout (d_model -> bottleneck_dim -> d_model)
      -> Output head (d_model -> 8 classes)

Logged quantities per forward pass:
    kl_list  : list of scalar tensors, VIB KL divergence
    rcr_list : list of scalar tensors, recurrence convergence rate ||H_t - H_{t-1}||
    h_list   : list of [B, L, d_model] tensors, hidden state at each step
    z_final  : [B, L, bottleneck_dim], final bottleneck representation

Author: Puzhi YU
Date:   January 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict


# ──────────────────────────────────────────────
# 1. Sinusoidal Positional Encoding
# ──────────────────────────────────────────────

def _sinusoidal_pe(L: int, d_model: int) -> Tensor:
    """Standard sinusoidal positional encoding, shape [1, L, d_model].
    Fixed (non-learnable), registered as a buffer."""
    pe  = torch.zeros(1, L, d_model)
    pos = torch.arange(L, dtype=torch.float).unsqueeze(1)          # [L, 1]
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * (-math.log(10000.0) / d_model)
    )                                                                # [d_model/2]
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


# ──────────────────────────────────────────────
# 2. Variational Information Bottleneck (VIB)
# ──────────────────────────────────────────────

class VIBLayer(nn.Module):
    """Variational Information Bottleneck applied to the full sequence.

    Compresses [B, L, d_model] -> [B, L, bottleneck_dim] -> [B, L, d_model]
    via the reparameterization trick.

    Training: z = mu + eps * sigma  (stochastic, applies forgetting pressure)
    Inference: z = mu               (deterministic, for visualization / probing)

    Closed-form KL:
        KL(N(mu, sigma^2) || N(0, I)) = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    """

    def __init__(self, d_model: int, bottleneck_dim: int):
        super().__init__()
        self.mu_head     = nn.Linear(d_model, bottleneck_dim)
        self.logvar_head = nn.Linear(d_model, bottleneck_dim)
        self.decoder     = nn.Linear(bottleneck_dim, d_model)

    def forward(self, h: Tensor):
        """
        Args:
            h: [B, L, d_model]
        Returns:
            h_out : [B, L, d_model]         reconstructed hidden state
            kl    : scalar                   KL divergence (loss term)
            z     : [B, L, bottleneck_dim]   bottleneck representation
        """
        mu     = self.mu_head(h)                    # [B, L, bottleneck_dim]
        logvar = self.logvar_head(h).clamp(-10, 10) # clamp for numerical stability

        if self.training:
            std = torch.exp(0.5 * logvar)
            z   = mu + torch.randn_like(std) * std
        else:
            z = mu  # deterministic at inference

        # Analytical KL, averaged over B, L, bottleneck_dim -> scalar
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()

        h_out = self.decoder(z)   # [B, L, d_model]
        return h_out, kl, z


# ──────────────────────────────────────────────
# 3. Main Model: RecurrentTransformer
# ──────────────────────────────────────────────

class RecurrentTransformer(nn.Module):
    """Minimalist weight-sharing Recurrent Transformer with VIB bottleneck.

    Design principles:
      - Single Transformer layer with weights shared across T recurrent steps
      - Forces rule learning through "thinking time" (recurrence) rather than
        parameter capacity (depth)
      - VIB applied at the readout stage to impose information compression
      - Causal mask ensures position t can only attend to positions 0..t

    Args:
        d_model:        hidden dimension (default 32)
        nhead:          number of attention heads (must divide d_model, default 4)
        bottleneck_dim: VIB bottleneck dimension (default 2, same as input dim)
        T:              recurrent steps (default 8)
        L:              sequence length (default 10)
        n_classes:      output classes (default 8, one per octagon vertex)
        dropout:        dropout rate (default 0.0, not needed for this model size)
    """

    def __init__(
        self,
        d_model:        int   = 32,
        nhead:          int   = 4,
        bottleneck_dim: int   = 2,
        T:              int   = 8,
        L:              int   = 10,
        n_classes:      int   = 8,
        dropout:        float = 0.0,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}"

        self.d_model        = d_model
        self.bottleneck_dim = bottleneck_dim
        self.T              = T
        self.L              = L

        # Input embedding: 2D geometric coordinates -> d_model
        self.embedding = nn.Linear(2, d_model)

        # Fixed sinusoidal positional encoding (not learnable)
        self.register_buffer('pos_enc', _sinusoidal_pe(L, d_model))

        # Shared-weight Transformer layer (same weights across all T steps)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,   # 32 -> 128 -> 32
            dropout=dropout,
            batch_first=True,
            norm_first=True,               # Pre-LayerNorm for training stability
        )

        # Variational information bottleneck (readout compression)
        self.vib = VIBLayer(d_model, bottleneck_dim)

        # Prediction head
        self.output_head = nn.Linear(d_model, n_classes)

        # Causal mask (generated once, reused across forward calls)
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
        )

    def forward(self, x: Tensor) -> Dict:
        """
        Args:
            x: [B, L, 2]  geometric coordinate sequence

        Returns dict with keys:
            logits   : [B, L, 8]              next-vertex prediction at each position
            kl_list  : list of scalar tensors  VIB KL divergence
            rcr_list : list of scalar tensors  recurrence convergence rate per step
            h_list   : list of [B, L, d_model] hidden states (incl. H_0, for probing)
            z_final  : [B, L, bottleneck_dim]  final bottleneck representation
        """
        B, L, _ = x.shape

        # Embedding + positional encoding
        H = self.embedding(x) + self.pos_enc      # [B, L, d_model]

        kl_list  : List[Tensor] = []
        rcr_list : List[Tensor] = []
        h_list   : List[Tensor] = [H.detach().cpu()]

        # Recurrent loop: T steps with shared weights, no noise injection inside.
        # Keeping the loop clean avoids noise accumulation across T steps,
        # which would otherwise cause Posterior Collapse.
        for _ in range(self.T):
            H_prev = H

            # Transformer step (shared weights, causal mask)
            H = self.transformer_layer(H, src_mask=self.causal_mask)

            # Track recurrence convergence rate (dynamics only, no noise here)
            rcr = (H - H_prev).norm(dim=-1).mean()
            rcr_list.append(rcr.detach())
            h_list.append(H.detach().cpu())

        # VIB applied at readout stage (after all T recurrent steps).
        # This models the GNW "broadcast bottleneck": compression acts on the
        # final result of recurrent computation, not on intermediate steps.
        H_out, kl, z_final = self.vib(H)
        kl_list.append(kl)   # single KL value from readout stage

        # Output head: independent next-vertex prediction at each position
        logits = self.output_head(H_out)    # [B, L, 8]

        return {
            'logits'  : logits,
            'kl_list' : kl_list,    # len=1, readout-stage KL
            'rcr_list': rcr_list,   # len=T, per-step convergence rate
            'h_list'  : h_list,     # len=T+1, including initial embedding H_0
            'z_final' : z_final.detach().cpu(),   # [B, L, bottleneck_dim]
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# 4. Quick self-test
# ──────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(0)

    model = RecurrentTransformer(d_model=32, nhead=4, bottleneck_dim=2, T=8, L=10)
    print(f"Total parameters: {model.count_parameters():,}")

    x = torch.randn(4, 10, 2)   # B=4, L=10
    out = model(x)

    print(f"logits  : {tuple(out['logits'].shape)}")
    print(f"kl_list : {len(out['kl_list'])} steps, first={out['kl_list'][0].item():.4f}")
    print(f"rcr_list: {len(out['rcr_list'])} steps, first={out['rcr_list'][0].item():.4f}")
    print(f"h_list  : {len(out['h_list'])} states, each {tuple(out['h_list'][0].shape)}")
    print(f"z_final : {tuple(out['z_final'].shape)}")

    # Verify T=1 baseline (single recurrent step)
    model_t1 = RecurrentTransformer(T=1)
    out_t1 = model_t1(x)
    print(f"\n[T=1 baseline] kl_list: {len(out_t1['kl_list'])} step")

    print("\n[OK] model.py self-test passed")
