"""
Internal World Model (Stub)
==========================

Lightweight internal world model for autonomous planning and simulation.
Reward-free: used only for consistency and counterfactual rollouts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class InternalWorldModel(nn.Module):
    def __init__(self, feature_dim: int, horizon: int = 50):
        super().__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        
        self.dynamics = nn.GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=1, batch_first=True)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, current_features: torch.Tensor, proposed_actions: torch.Tensor, horizon: Optional[int] = None) -> Dict[str, torch.Tensor]:
        B = current_features.size(0)
        T = horizon or self.horizon
        # simple rollout: repeat features and evolve with GRU
        x = current_features.unsqueeze(1).expand(B, T, -1)
        pred, _ = self.dynamics(x)
        pred = self.project(pred)
        return {
            'predictions': pred,          # [B, T, D]
            'final_state': pred[:, -1],   # [B, D]
        }
