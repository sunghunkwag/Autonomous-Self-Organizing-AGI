"""
Consistency-Based Learner (Upgraded)
===================================

Upgrades over the stub:
- Uses CPC signal from world model as a self-supervised target (no labels)
- EMA teacher for stability (BYOL-style) without contrastive negatives
- Fast, batched, purely feed-forward updates (GPU-parallel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class EMA(nn.Module):
    def __init__(self, module: nn.Module, decay: float = 0.99):
        super().__init__()
        self.module = module
        self.decay = decay
        # freeze
        for p in self.module.parameters():
            p.requires_grad = False
    @torch.no_grad()
    def update(self, online: nn.Module):
        for p_t, p_o in zip(self.module.parameters(), online.parameters()):
            p_t.data.mul_(self.decay).add_(p_o.data, alpha=1-self.decay)

class Adapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class ConsistencyBasedLearner(nn.Module):
    def __init__(self, feature_dim: int, threshold: float = 0.8):
        super().__init__()
        self.feature_dim = feature_dim
        self.threshold = threshold
        
        self.online = Adapter(feature_dim)
        self.target = EMA(Adapter(feature_dim), decay=0.995)
        self.register_buffer('consistency_score', torch.tensor(0.0))
    
    def forward(self, observations: torch.Tensor, predictions: torch.Tensor, internal_state: torch.Tensor, cpc_signal: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Use first-step prediction to align
        pred_now = predictions[:, 0]
        proj_obs = self.online(observations)
        with torch.no_grad():
            self.target.update(self.online)
            proj_pred = self.target.module(pred_now)
        align = F.cosine_similarity(proj_obs, proj_pred, dim=-1)
        self.consistency_score = 0.9 * self.consistency_score + 0.1 * align.mean().detach()
        
        # Optional CPC guidance: push alignment when CPC is strong
        if cpc_signal is not None:
            weight = torch.sigmoid(cpc_signal).unsqueeze(-1)
            proj_obs = proj_obs * (1 + 0.1*weight)
        
        adjustments = proj_obs  # suggest moving internal state toward consistent subspace
        return {
            'alignment': align,
            'adjustments': adjustments,
            'consistency_score': self.consistency_score.unsqueeze(0)
        }
