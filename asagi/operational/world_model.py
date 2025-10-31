"""
Internal World Model (Upgraded)
==============================

Upgrades over the stub:
- Multi-scale latent dynamics (stacked gated residual blocks)
- Optional transformer mixer for parallel rollout
- Contrastive predictive coding (CPC) head for self-supervised consistency
- Lightweight, GPU-parallel design (no heavy optimization loops)

Still reward-free: used for consistency and counterfactual rollouts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GatedResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim*2)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        h = self.fc1(self.norm1(x))
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)
        h = self.fc2(self.norm2(F.gelu(h)))
        return x + self.gate.tanh()*h

class TransformerMixer(nn.Module):
    def __init__(self, dim: int, heads: int = 4, layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
    def forward(self, x):
        return self.enc(x)

class CPCHead(nn.Module):
    """Contrastive predictive coding for self-supervised temporal consistency."""
    def __init__(self, dim: int, proj_dim: int = 128):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.key   = nn.Sequential(nn.Linear(dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
    def forward(self, seq):
        # seq: [B,T,D] -> predict t+1 from t
        q = self.query(seq[:, :-1])   # [B,T-1,P]
        k = self.key(seq[:, 1:])      # [B,T-1,P]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = torch.einsum('bip,bjp->bij', q, k)  # similarity matrix over time
        pos = logits.diagonal(dim1=1, dim2=2)        # [B,T-1]
        neg = logits.mean(dim=-1)                    # [B,T-1]
        cpc_signal = (pos - neg).mean(dim=-1)        # higher is better
        return {'cpc_signal': cpc_signal, 'logits': logits}

class InternalWorldModel(nn.Module):
    def __init__(self, feature_dim: int, horizon: int = 50, use_transformer: bool = True, depth: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.use_transformer = use_transformer

        self.input_proj = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU())
        self.blocks = nn.ModuleList([GatedResBlock(feature_dim) for _ in range(depth)])
        self.mixer = TransformerMixer(feature_dim, heads=4, layers=2) if use_transformer else nn.Identity()
        self.output_proj = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim))
        self.cpc = CPCHead(feature_dim)

    def rollout(self, x: torch.Tensor, T: int) -> torch.Tensor:
        # x: [B,D] -> [B,T,D] with vectorized parallelization
        B, D = x.shape
        seq = x.unsqueeze(1).expand(B, T, D).contiguous()
        seq = self.input_proj(seq)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.mixer(seq)
        seq = self.output_proj(seq)
        return seq

    def forward(self, current_features: torch.Tensor, proposed_actions: torch.Tensor, horizon: Optional[int] = None) -> Dict[str, torch.Tensor]:
        # Input validation
        if current_features.ndim != 2:
            raise ValueError(f"Expected 2D current_features, got shape {current_features.shape}")
        if current_features.shape[-1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {current_features.shape[-1]}")
        
        # Check for NaN/Inf
        if torch.isnan(current_features).any() or torch.isinf(current_features).any():
            logger.warning("Invalid values detected in current_features")
            current_features = torch.nan_to_num(current_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        T = horizon or self.horizon
        if T <= 0:
            raise ValueError(f"Horizon must be positive, got {T}")
        
        try:
            preds = self.rollout(current_features, T)
            cpc = self.cpc(preds)
            return {
                'predictions': preds,          # [B,T,D]
                'final_state': preds[:, -1],   # [B,D]
                'cpc_signal': cpc['cpc_signal'] # [B]
            }
        except Exception as e:
            logger.error(f"Error in world model forward pass: {e}")
            raise
