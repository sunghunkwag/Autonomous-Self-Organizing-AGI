"""
Causal Reasoning Module (GNN-based)
===================================

Graph Neural Network-based module for learning and reasoning about
causal relationships between variables. Reward-free and self-supervised
via structure sparsity and stability priors.

Key Features:
- Attention-based causal structure discovery
- Message passing GNN for causal propagation
- Intervention simulation for causal discovery
- Counterfactual reasoning
- Causal effect estimation
- Batched tensor ops; optional sparse approximations for scalability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class CausalGraphEncoder(nn.Module):
    """
    Encodes observations into a causal graph structure.
    Uses attention-style pair scoring to estimate directed edges.
    """
    def __init__(self, feature_dim: int, num_variables: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        
        # Variable extraction (project global features to N variable slots)
        self.variable_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_variables * hidden_dim)
        )
        
        # Edge scoring network (shared for all pairs)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.sparsity_temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = features.shape[0]
        # Extract variables
        var_flat = self.variable_extractor(features)
        variables = var_flat.view(B, self.num_variables, self.hidden_dim)
        
        # Vectorized pairwise edge scoring
        # Build all i,j pairs
        v_i = variables.unsqueeze(2).expand(B, self.num_variables, self.num_variables, self.hidden_dim)
        v_j = variables.unsqueeze(1).expand(B, self.num_variables, self.num_variables, self.hidden_dim)
        pair = torch.cat([v_i, v_j], dim=-1)  # [B,N,N,2H]
        scores = self.edge_scorer(pair).squeeze(-1)  # [B,N,N]
        
        # Mask self-edges and apply sigmoid with temperature
        eye = torch.eye(self.num_variables, device=features.device).unsqueeze(0)
        scores = scores.masked_fill(eye.bool(), float('-inf'))
        adjacency = torch.sigmoid(scores / (self.sparsity_temperature.abs() + 1e-6))
        
        return variables, adjacency

class GraphConvolutionLayer(nn.Module):
    """Graph convolution layer for message passing on causal graphs."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.update = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        B, N, D = node_features.shape
        # Vectorized message passing: for all i<-j
        v_i = node_features.unsqueeze(2).expand(B, N, N, D)
        v_j = node_features.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([v_j, v_i], dim=-1)  # messages j->i
        msg_all = self.message(pair)  # [B,N,N,out]
        # Weight by adjacency (send j->i)
        weights = adjacency.unsqueeze(-1)  # [B,N,N,1]
        agg = (msg_all * weights).sum(dim=2)  # sum over j: [B,N,out]
        updated = self.update(torch.cat([node_features, agg], dim=-1))
        return updated

class InterventionSimulator(nn.Module):
    """Simulates interventions do(X=x) on the learned variables."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.effect = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, variables: torch.Tensor, adjacency: torch.Tensor, idx: int, value: torch.Tensor) -> torch.Tensor:
        B, N, D = variables.shape
        z = variables.clone()
        z[:, idx] = value
        # Propagate effect to children of idx
        # children mask: adjacency[idx->i]
        child_w = adjacency[:, idx]  # [B,N]
        v_src = z[:, idx]            # [B,D]
        v_src_exp = v_src.unsqueeze(1).expand(B, N, D)
        pair = torch.cat([v_src_exp, z], dim=-1)
        delta = self.effect(pair)
        z = z + delta * child_w.unsqueeze(-1)
        return z

class CounterfactualReasoner(nn.Module):
    """Counterfactual reasoning: compare factual vs counterfactual contexts."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, factual: torch.Tensor, counterfactual: torch.Tensor, query_idx: int) -> torch.Tensor:
        # factual/counterfactual: [B,N,D]
        f = factual[:, query_idx]
        cf = counterfactual[:, query_idx]
        ctx = (factual - counterfactual).mean(dim=1)
        return self.net(torch.cat([f, cf, ctx], dim=-1))

class CausalReasoningModule(nn.Module):
    """End-to-end causal reasoning with structure learning and interventions."""
    def __init__(self, feature_dim: int, decision_dim: int, num_variables: int = 8, hidden_dim: int = 128, num_gnn_layers: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.decision_dim = decision_dim
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(feature_dim + decision_dim, feature_dim)
        self.graph_encoder = CausalGraphEncoder(feature_dim, num_variables, hidden_dim)
        self.gnn_layers = nn.ModuleList([GraphConvolutionLayer(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
        self.intervention = InterventionSimulator(hidden_dim)
        self.counterfactual = CounterfactualReasoner(hidden_dim)
        self.effect_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.output_proj = nn.Linear(num_variables * hidden_dim, 64)
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor, world_simulation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = observations.shape[0]
        x = self.input_proj(torch.cat([observations, actions], dim=-1))
        vars0, adj = self.graph_encoder(x)
        vars_t = vars0
        for layer in self.gnn_layers:
            vars_t = layer(vars_t, adj)
        
        # Pairwise causal effects (vectorized)
        v_i = vars_t.unsqueeze(2).expand(B, self.num_variables, self.num_variables, self.hidden_dim)
        v_j = vars_t.unsqueeze(1).expand(B, self.num_variables, self.num_variables, self.hidden_dim)
        pair = torch.cat([v_i, v_j], dim=-1)
        eff = self.effect_estimator(pair).squeeze(-1)
        eff = eff.masked_fill(torch.eye(self.num_variables, device=observations.device).unsqueeze(0).bool(), 0.0)
        
        # Example intervention on variable 0
        z_val = torch.randn(B, self.hidden_dim, device=observations.device)
        intervened = self.intervention(vars_t, adj, 0, z_val)
        
        # Counterfactual: small action perturbation
        a_cf = actions + 0.1 * torch.randn_like(actions)
        x_cf = self.input_proj(torch.cat([observations, a_cf], dim=-1))
        vars_cf, _ = self.graph_encoder(x_cf)
        cf_out = self.counterfactual(vars_t, vars_cf, query_idx=0)
        
        causal_repr = self.output_proj(vars_t.flatten(1))
        
        return {
            'causal_graph': adj,
            'variables': vars_t,
            'causal_effects': eff,
            'intervention_effects': intervened,
            'counterfactual_outcome': cf_out,
            'causal_representation': causal_repr,
            'graph_sparsity': adj.mean()
        }
    
    def discover_causal_structure(self, observations_sequence: torch.Tensor) -> torch.Tensor:
        B, T, D = observations_sequence.shape
        graphs = []
        for t in range(T):
            _, adj = self.graph_encoder(observations_sequence[:, t])
            graphs.append(adj)
        return torch.stack(graphs, dim=0).mean(dim=0)
    
    def estimate_causal_effect(self, observations: torch.Tensor, cause_idx: int, effect_idx: int) -> torch.Tensor:
        vars0, adj = self.graph_encoder(observations)
        vt = vars0
        for layer in self.gnn_layers:
            vt = layer(vt, adj)
        pair = torch.cat([vt[:, cause_idx], vt[:, effect_idx]], dim=-1)
        return self.effect_estimator(pair).squeeze(-1)
