"""
Intrinsic Signal Synthesizer
===========================

The core of reward-free artificial intelligence. This module generates
intrinsic motivation signals based purely on information-theoretic principles,
without any external rewards, task objectives, or loss functions.

Key Principles:
- Predictive Dissonance: KL-divergence between predictions and reality
- Compression Gain: Minimum Description Length (MDL) improvements  
- Uncertainty Reduction: Epistemic and aleatoric uncertainty changes
- Novelty Topology: Manifold structure evolution in representation space

These signals drive autonomous goal generation and learning without
any external supervision or reward engineering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import math

class PredictiveDissonanceEstimator(nn.Module):
    """
    Measures the mismatch between model predictions and observations.
    
    This is the primary curiosity signal - areas where our predictions
    fail indicate opportunities for learning and exploration.
    """
    
    def __init__(self, feature_dim: int, history_length: int = 100):
        super().__init__()
        self.feature_dim = feature_dim
        self.history_length = history_length
        
        # Predictive model for next observations
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2), 
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # History buffer for temporal patterns
        self.register_buffer(
            'observation_history', 
            torch.zeros(history_length, feature_dim)
        )
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, current_obs: torch.Tensor, 
                previous_obs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute predictive dissonance between predicted and actual observations.
        
        Args:
            current_obs: Current observation [B, feature_dim]
            previous_obs: Previous observation for temporal prediction [B, feature_dim]
            
        Returns:
            Dictionary containing dissonance metrics
        """
        batch_size = current_obs.shape[0]
        
        if previous_obs is None:
            # Use history buffer
            if self.history_ptr > 0:
                previous_obs = self.observation_history[self.history_ptr - 1].unsqueeze(0).expand(batch_size, -1)
            else:
                previous_obs = torch.zeros_like(current_obs)
        
        # Generate prediction
        predicted_features = self.predictor.forward(previous_obs)
        predicted_obs = predicted_features
        
        # Estimate prediction uncertainty  
        uncertainty = self.uncertainty_head(predicted_features)
        
        # Compute various dissonance metrics
        # 1. Mean Squared Error (basic prediction error)
        mse_dissonance = F.mse_loss(predicted_obs, current_obs, reduction='none').mean(dim=-1)
        
        # 2. KL Divergence approximation (assuming Gaussian distributions)
        pred_var = uncertainty.squeeze(-1) + 1e-8
        kl_dissonance = self._approximate_kl_divergence(
            predicted_obs, current_obs, pred_var
        )
        
        # 3. Cosine dissimilarity (directional differences)
        cosine_sim = F.cosine_similarity(predicted_obs, current_obs, dim=-1)
        cosine_dissonance = 1.0 - cosine_sim
        
        # 4. Information-theoretic surprise (negative log-likelihood)
        log_likelihood = self._compute_log_likelihood(
            predicted_obs, current_obs, pred_var
        )
        surprise = -log_likelihood
        
        # Update history buffer
        self._update_history(current_obs)
        
        return {
            'mse_dissonance': mse_dissonance,
            'kl_dissonance': kl_dissonance,
            'cosine_dissonance': cosine_dissonance,
            'surprise': surprise,
            'uncertainty': uncertainty.squeeze(-1),
            'total_dissonance': (mse_dissonance + kl_dissonance + cosine_dissonance + surprise) / 4.0
        }
    
    def _approximate_kl_divergence(self, pred_mean: torch.Tensor, 
                                  true_obs: torch.Tensor,
                                  pred_var: torch.Tensor) -> torch.Tensor:
        """Approximate KL divergence assuming Gaussian distributions."""
        # Assume true observations have unit variance (normalized)
        true_var = torch.ones_like(pred_var)
        
        # KL(true || pred) = 0.5 * (log(var_pred/var_true) + var_true/var_pred + (mean_diff)^2/var_pred - 1)
        mean_diff_sq = (true_obs - pred_mean).pow(2).mean(dim=-1)
        
        kl_div = 0.5 * (
            torch.log(pred_var / true_var) + 
            true_var / pred_var + 
            mean_diff_sq / pred_var - 1.0
        )
        
        return kl_div.clamp(min=0.0)  # KL divergence is always non-negative
    
    def _compute_log_likelihood(self, pred_mean: torch.Tensor,
                               true_obs: torch.Tensor, 
                               pred_var: torch.Tensor) -> torch.Tensor:
        """Compute log-likelihood under predicted Gaussian distribution."""
        diff_sq = (true_obs - pred_mean).pow(2).sum(dim=-1)
        log_likelihood = -0.5 * (
            torch.log(2 * math.pi * pred_var) + 
            diff_sq / pred_var
        )
        return log_likelihood
    
    def _update_history(self, observation: torch.Tensor):
        """Update circular history buffer."""
        # Take first sample from batch for history
        obs_sample = observation[0].detach()
        
        ptr = self.history_ptr.item()
        self.observation_history[ptr] = obs_sample
        self.history_ptr[0] = (ptr + 1) % self.history_length

class CompressionGainAnalyzer(nn.Module):
    """
    Analyzes potential gains in compression when incorporating new information.
    
    Based on Minimum Description Length (MDL) principle - prefers models
    that compress data well while maintaining predictive accuracy.
    """
    
    def __init__(self, feature_dim: int, codebook_size: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size
        
        # Vector quantization for compression estimation
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
        
        # Codebook for quantization
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, feature_dim // 4) * 0.1
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim // 4, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
        
        # Model complexity tracker
        self.register_buffer('model_complexity', torch.zeros(1))
        self.register_buffer('data_likelihood', torch.zeros(1))
        
    def forward(self, features: torch.Tensor, 
                new_hypothesis: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute compression gain from incorporating new information.
        
        Args:
            features: Current feature representations [B, feature_dim]
            new_hypothesis: New information to potentially incorporate [B, feature_dim]
            
        Returns:
            Dictionary containing compression metrics
        """
        batch_size = features.shape[0]
        
        # Encode to compressed representation
        encoded = self.encoder(features)
        
        # Vector quantization
        distances = torch.cdist(encoded, self.codebook)
        quantized_indices = distances.argmin(dim=-1)
        quantized = self.codebook[quantized_indices]
        
        # Decode back
        reconstructed = self.decoder(quantized)
        
        # Compute current compression metrics
        reconstruction_error = F.mse_loss(reconstructed, features, reduction='none').mean(dim=-1)
        
        # Estimate description length (bits needed to encode)
        # Based on quantization indices and reconstruction error
        index_bits = math.log2(self.codebook_size)  # Bits per index
        error_bits = self._estimate_error_bits(reconstruction_error)
        total_bits = index_bits + error_bits
        
        compression_ratio = self.feature_dim * 32 / total_bits  # Assuming 32-bit floats
        
        # If new hypothesis provided, estimate improvement
        compression_gain = torch.zeros(batch_size, device=features.device)
        
        if new_hypothesis is not None:
            # Compute compression with new information incorporated
            combined_features = torch.cat([features, new_hypothesis], dim=-1)
            # Project back to original dimension
            combined_proj = nn.Linear(features.shape[-1] + new_hypothesis.shape[-1], 
                                    features.shape[-1], device=features.device)
            combined_features = combined_proj(combined_features)
            
            # Recompute compression metrics
            new_encoded = self.encoder(combined_features)
            new_distances = torch.cdist(new_encoded, self.codebook)
            new_indices = new_distances.argmin(dim=-1)
            new_quantized = self.codebook[new_indices]
            new_reconstructed = self.decoder(new_quantized)
            
            new_error = F.mse_loss(new_reconstructed, combined_features, reduction='none').mean(dim=-1)
            new_error_bits = self._estimate_error_bits(new_error)
            new_total_bits = index_bits + new_error_bits
            new_compression_ratio = combined_features.shape[-1] * 32 / new_total_bits
            
            # Compression gain is improvement in compression ratio
            compression_gain = new_compression_ratio - compression_ratio
        
        return {
            'reconstruction_error': reconstruction_error,
            'compression_ratio': compression_ratio,
            'compression_gain': compression_gain,
            'description_length': total_bits,
            'quantization_indices': quantized_indices
        }
    
    def _estimate_error_bits(self, error: torch.Tensor) -> torch.Tensor:
        """Estimate bits needed to encode reconstruction error."""
        # Approximate using negative log probability
        # Assume errors follow a Laplace distribution
        scale = error.mean() + 1e-8
        log_prob = -torch.abs(error) / scale - torch.log(2 * scale)
        bits = -log_prob / math.log(2)
        return bits

class UncertaintyReductionTracker(nn.Module):
    """
    Tracks reduction in both epistemic (model) and aleatoric (data) uncertainty.
    
    Uncertainty reduction indicates learning progress and guides exploration
    toward areas where knowledge can be most improved.
    """
    
    def __init__(self, feature_dim: int, num_ensembles: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_ensembles = num_ensembles
        
        # Ensemble of predictors for epistemic uncertainty
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(num_ensembles)
        ])
        
        # Aleatoric uncertainty estimator
        self.aleatoric_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()
        )
        
        # History of uncertainties
        self.register_buffer('uncertainty_history', torch.zeros(100))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute epistemic and aleatoric uncertainty and their reduction.
        
        Args:
            features: Input features [B, feature_dim]
            
        Returns:
            Dictionary containing uncertainty metrics
        """
        batch_size = features.shape[0]
        
        # Epistemic uncertainty via ensemble variance
        ensemble_predictions = []
        for predictor in self.ensemble:
            pred = predictor(features)
            ensemble_predictions.append(pred)
        
        ensemble_stack = torch.stack(ensemble_predictions, dim=0)  # [num_ensembles, B, feature_dim]
        ensemble_mean = ensemble_stack.mean(dim=0)  # [B, feature_dim]
        ensemble_var = ensemble_stack.var(dim=0)  # [B, feature_dim]
        
        epistemic_uncertainty = ensemble_var.mean(dim=-1)  # [B]
        
        # Aleatoric uncertainty
        aleatoric_uncertainty = self.aleatoric_estimator(features).squeeze(-1)  # [B]
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Compute uncertainty reduction
        current_avg_uncertainty = total_uncertainty.mean()
        
        # Get historical uncertainty for comparison
        if self.history_ptr > 0:
            historical_uncertainty = self.uncertainty_history[:self.history_ptr].mean()
            uncertainty_reduction = historical_uncertainty - current_avg_uncertainty
        else:
            uncertainty_reduction = torch.zeros_like(current_avg_uncertainty)
        
        # Update history
        ptr = self.history_ptr.item()
        if ptr < 100:
            self.uncertainty_history[ptr] = current_avg_uncertainty.detach()
            self.history_ptr[0] = ptr + 1
        else:
            # Circular buffer
            self.uncertainty_history = torch.roll(self.uncertainty_history, -1)
            self.uncertainty_history[-1] = current_avg_uncertainty.detach()
        
        return {
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'uncertainty_reduction': uncertainty_reduction.expand(batch_size),
            'ensemble_predictions': ensemble_stack
        }

class NoveltyTopologyAnalyzer(nn.Module):
    """
    Analyzes changes in the topology of representation manifolds.
    
    Detects when new experiences create novel structures in the learned
    representation space, indicating discovery of new patterns or concepts.
    """
    
    def __init__(self, feature_dim: int, manifold_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.manifold_dim = manifold_dim
        
        # Manifold projection network
        self.manifold_projector = nn.Sequential(
            nn.Linear(feature_dim, manifold_dim * 2),
            nn.LayerNorm(manifold_dim * 2),
            nn.GELU(),
            nn.Linear(manifold_dim * 2, manifold_dim)
        )
        
        # Topology analyzer
        self.topology_encoder = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim // 2),
            nn.GELU(),
            nn.Linear(manifold_dim // 2, 16)  # Topology signature
        )
        
        # History of manifold representations
        self.register_buffer('manifold_history', torch.zeros(1000, manifold_dim))
        self.register_buffer('history_size', torch.zeros(1, dtype=torch.long))
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze novelty in representational topology.
        
        Args:
            features: Input features [B, feature_dim]
            
        Returns:
            Dictionary containing topology novelty metrics
        """
        batch_size = features.shape[0]
        
        # Project to manifold space
        manifold_repr = self.manifold_projector(features)
        
        # Compute topology signature
        topology_sig = self.topology_encoder(manifold_repr)
        
        # Analyze novelty compared to historical manifold
        novelty_scores = torch.zeros(batch_size, device=features.device)
        
        if self.history_size > 10:  # Need some history for comparison
            hist_size = min(self.history_size.item(), 1000)
            historical_manifold = self.manifold_history[:hist_size]
            
            # Compute distances to historical points
            distances = torch.cdist(manifold_repr, historical_manifold)
            min_distances = distances.min(dim=-1)[0]  # Closest historical point
            
            # Novelty is distance to nearest historical representation
            novelty_scores = min_distances
            
            # Additional topology-based novelty
            historical_topology = self.topology_encoder(
                historical_manifold[-min(50, hist_size):].mean(dim=0, keepdim=True)
            )
            
            topology_novelty = torch.norm(
                topology_sig - historical_topology, dim=-1
            )
            
            novelty_scores = novelty_scores + topology_novelty
        
        # Update history
        self._update_manifold_history(manifold_repr)
        
        # Compute additional topology metrics
        intrinsic_dimensionality = self._estimate_intrinsic_dimension(manifold_repr)
        local_curvature = self._estimate_local_curvature(manifold_repr)
        
        return {
            'novelty_scores': novelty_scores,
            'topology_signature': topology_sig,
            'manifold_representation': manifold_repr,
            'intrinsic_dimensionality': intrinsic_dimensionality,
            'local_curvature': local_curvature
        }
    
    def _update_manifold_history(self, manifold_repr: torch.Tensor):
        """Update history of manifold representations."""
        # Add first sample from batch to history
        new_repr = manifold_repr[0].detach()
        
        hist_size = self.history_size.item()
        if hist_size < 1000:
            self.manifold_history[hist_size] = new_repr
            self.history_size[0] = hist_size + 1
        else:
            # Circular buffer - replace oldest
            self.manifold_history = torch.roll(self.manifold_history, -1, dims=0)
            self.manifold_history[-1] = new_repr
    
    def _estimate_intrinsic_dimension(self, manifold_repr: torch.Tensor) -> torch.Tensor:
        """Estimate intrinsic dimensionality using correlation dimension."""
        # Simplified estimate based on eigenvalue spectrum
        if manifold_repr.shape[0] < 2:
            return torch.tensor(0.0, device=manifold_repr.device)
        
        # Compute covariance matrix
        centered = manifold_repr - manifold_repr.mean(dim=0, keepdim=True)
        cov = torch.matmul(centered.T, centered) / (manifold_repr.shape[0] - 1)
        
        # Eigenvalues
        eigenvals = torch.linalg.eigvals(cov).real
        eigenvals = eigenvals[eigenvals > 0]  # Only positive eigenvalues
        
        if len(eigenvals) == 0:
            return torch.tensor(0.0, device=manifold_repr.device)
        
        # Participation ratio as dimension estimate
        eigenvals_norm = eigenvals / eigenvals.sum()
        participation_ratio = 1.0 / (eigenvals_norm ** 2).sum()
        
        return participation_ratio
    
    def _estimate_local_curvature(self, manifold_repr: torch.Tensor) -> torch.Tensor:
        """Estimate local curvature of manifold."""
        if manifold_repr.shape[0] < 3:
            return torch.zeros(manifold_repr.shape[0], device=manifold_repr.device)
        
        # Compute pairwise distances
        distances = torch.cdist(manifold_repr, manifold_repr)
        
        # For each point, find k-nearest neighbors
        k = min(5, manifold_repr.shape[0] - 1)
        _, nearest_indices = distances.topk(k + 1, dim=-1, largest=False)
        nearest_indices = nearest_indices[:, 1:]  # Exclude self
        
        curvatures = []
        for i in range(manifold_repr.shape[0]):
            neighbors = manifold_repr[nearest_indices[i]]
            center = manifold_repr[i]
            
            # Compute local covariance
            centered_neighbors = neighbors - center.unsqueeze(0)
            local_cov = torch.matmul(centered_neighbors.T, centered_neighbors)
            
            # Curvature approximated by condition number
            try:
                eigenvals = torch.linalg.eigvals(local_cov).real
                eigenvals = eigenvals[eigenvals > 1e-6]
                if len(eigenvals) > 1:
                    condition_number = eigenvals.max() / eigenvals.min()
                    curvature = torch.log(condition_number)
                else:
                    curvature = torch.tensor(0.0, device=manifold_repr.device)
            except:
                curvature = torch.tensor(0.0, device=manifold_repr.device)
            
            curvatures.append(curvature)
        
        return torch.stack(curvatures)

class IntrinsicSignalSynthesizer(nn.Module):
    """
    Main synthesizer that combines all intrinsic motivation signals.
    
    This is the heart of the reward-free system - it generates motivation
    signals purely from internal dynamics without any external objectives.
    """
    
    def __init__(self, feature_dim: int, config: Optional[Dict] = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = config or {}
        
        # Initialize all signal generators
        self.dissonance_estimator = PredictiveDissonanceEstimator(feature_dim)
        self.compression_analyzer = CompressionGainAnalyzer(feature_dim)
        self.uncertainty_tracker = UncertaintyReductionTracker(feature_dim)
        self.novelty_analyzer = NoveltyTopologyAnalyzer(feature_dim)
        
        # Signal fusion network
        self.signal_fusion = nn.Sequential(
            nn.Linear(4, 16),  # 4 main signal types
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 4)  # Output combined signals
        )
        
        # Signal weights (learnable)
        self.signal_weights = nn.Parameter(torch.ones(4))
        
        # Global intrinsic motivation level
        self.register_buffer('motivation_level', torch.ones(1))
        
    def forward(self, features: torch.Tensor, 
                previous_features: Optional[torch.Tensor] = None,
                new_hypothesis: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Synthesize all intrinsic motivation signals.
        
        Args:
            features: Current features [B, feature_dim]
            previous_features: Previous features for temporal analysis
            new_hypothesis: Potential new information to incorporate
            
        Returns:
            Dictionary containing all intrinsic signals and combined motivation
        """
        # Generate individual signals
        dissonance_signals = self.dissonance_estimator(features, previous_features)
        compression_signals = self.compression_analyzer(features, new_hypothesis)
        uncertainty_signals = self.uncertainty_tracker(features)
        novelty_signals = self.novelty_analyzer(features)
        
        # Extract primary signals
        primary_signals = torch.stack([
            dissonance_signals['total_dissonance'],
            compression_signals['compression_gain'],
            uncertainty_signals['uncertainty_reduction'],
            novelty_signals['novelty_scores']
        ], dim=-1)  # [B, 4]
        
        # Normalize signals to [0, 1] range
        normalized_signals = torch.sigmoid(primary_signals)
        
        # Apply learnable weights
        weighted_signals = normalized_signals * F.softmax(self.signal_weights, dim=0)
        
        # Fuse signals through neural network
        fused_signals = self.signal_fusion(weighted_signals)
        
        # Compute overall intrinsic motivation
        intrinsic_motivation = fused_signals.mean(dim=-1)  # [B]
        
        # Update global motivation level (exponential moving average)
        global_motivation = intrinsic_motivation.mean()
        self.motivation_level = 0.99 * self.motivation_level + 0.01 * global_motivation
        
        return {
            # Individual signal components
            'dissonance': dissonance_signals,
            'compression': compression_signals,
            'uncertainty': uncertainty_signals,
            'novelty': novelty_signals,
            
            # Combined signals
            'primary_signals': primary_signals,
            'normalized_signals': normalized_signals,
            'weighted_signals': weighted_signals,
            'fused_signals': fused_signals,
            
            # Overall motivation
            'intrinsic_motivation': intrinsic_motivation,
            'global_motivation_level': self.motivation_level.expand_as(intrinsic_motivation),
            
            # Signal weights for analysis
            'signal_weights': F.softmax(self.signal_weights, dim=0)
        }
    
    def get_curiosity_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate a curiosity map indicating where to explore next.
        
        Args:
            features: Current features [B, feature_dim]
            
        Returns:
            Curiosity map [B] indicating exploration priorities
        """
        signals = self.forward(features)
        
        # Combine uncertainty and novelty for exploration priority
        uncertainty_signal = signals['uncertainty']['total_uncertainty']
        novelty_signal = signals['novelty']['novelty_scores']
        
        curiosity_map = 0.6 * uncertainty_signal + 0.4 * novelty_signal
        
        return curiosity_map
    
    def detect_learning_opportunities(self, features: torch.Tensor, 
                                    threshold: float = 0.7) -> Dict[str, torch.Tensor]:
        """
        Detect specific learning opportunities based on signal analysis.
        
        Args:
            features: Current features [B, feature_dim]
            threshold: Threshold for opportunity detection
            
        Returns:
            Dictionary indicating different types of learning opportunities
        """
        signals = self.forward(features)
        
        opportunities = {
            'high_dissonance': signals['dissonance']['total_dissonance'] > threshold,
            'compression_potential': signals['compression']['compression_gain'] > threshold,
            'uncertainty_regions': signals['uncertainty']['total_uncertainty'] > threshold,
            'novel_topology': signals['novelty']['novelty_scores'] > threshold
        }
        
        # Overall learning opportunity score
        opportunity_scores = (
            signals['dissonance']['total_dissonance'] * 0.3 +
            signals['compression']['compression_gain'] * 0.2 +
            signals['uncertainty']['total_uncertainty'] * 0.3 +
            signals['novelty']['novelty_scores'] * 0.2
        )
        
        opportunities['overall_opportunity'] = opportunity_scores > threshold
        opportunities['opportunity_scores'] = opportunity_scores
        
        return opportunities

# Utility functions
def create_intrinsic_synthesizer(feature_dim: int, config: Optional[Dict] = None) -> IntrinsicSignalSynthesizer:
    """Factory function to create intrinsic signal synthesizer."""
    return IntrinsicSignalSynthesizer(feature_dim, config)

def analyze_motivation_patterns(signals_history: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """
    Analyze patterns in intrinsic motivation over time.
    
    Args:
        signals_history: List of signal dictionaries over time
        
    Returns:
        Analysis of motivation patterns
    """
    if not signals_history:
        return {}
    
    # Extract time series of each signal type
    dissonance_series = [s['dissonance']['total_dissonance'].mean().item() for s in signals_history]
    compression_series = [s['compression']['compression_gain'].mean().item() for s in signals_history]
    uncertainty_series = [s['uncertainty']['uncertainty_reduction'].mean().item() for s in signals_history]
    novelty_series = [s['novelty']['novelty_scores'].mean().item() for s in signals_history]
    
    return {
        'dissonance_trend': np.polyfit(range(len(dissonance_series)), dissonance_series, 1)[0],
        'compression_trend': np.polyfit(range(len(compression_series)), compression_series, 1)[0],
        'uncertainty_trend': np.polyfit(range(len(uncertainty_series)), uncertainty_series, 1)[0],
        'novelty_trend': np.polyfit(range(len(novelty_series)), novelty_series, 1)[0],
        
        'dissonance_volatility': np.std(dissonance_series),
        'compression_volatility': np.std(compression_series),
        'uncertainty_volatility': np.std(uncertainty_series),
        'novelty_volatility': np.std(novelty_series),
        
        'average_motivation': np.mean([s['intrinsic_motivation'].mean().item() for s in signals_history])
    }