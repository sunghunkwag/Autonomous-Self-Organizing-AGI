"""
Lightweight mode switches for experimental modules
- Consciousness module -> global workspace style aggregator
- Creative synthesizer -> retrieval-augmented recombination of skills
All kept lightweight, batched, and tensor-parallel to avoid compute blowups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_consciousness_module(dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim*3, dim), nn.LayerNorm(dim), nn.GELU(),
        nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, 32)
    )

def build_creative_synthesizer(dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim*2, dim), nn.LayerNorm(dim), nn.GELU(),
        nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim)
    )
