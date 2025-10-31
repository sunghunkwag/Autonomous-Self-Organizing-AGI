"""
Metrics Tracking System
=======================
Tracks and computes performance metrics for the ASAGI system.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import json
from pathlib import Path


class MetricsTracker:
    """Tracks metrics over time with statistical summaries."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.global_metrics: Dict[str, List[float]] = defaultdict(list)
        self.step = 0
        
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number (auto-increments if None)
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
            
        for name, value in metrics.items():
            # Convert tensor to float if needed
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.metrics[name].append(value)
            self.global_metrics[name].append(value)
    
    def get_current(self, name: str) -> Optional[float]:
        """Get the most recent value of a metric."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1]
        return None
    
    def get_mean(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """
        Get mean value of a metric over window.
        
        Args:
            name: Metric name
            window: Window size (uses default if None)
            
        Returns:
            Mean value or None if metric doesn't exist
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        
        values = list(self.metrics[name])
        if window is not None:
            values = values[-window:]
        
        return np.mean(values)
    
    def get_std(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """Get standard deviation of a metric over window."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        
        values = list(self.metrics[name])
        if window is not None:
            values = values[-window:]
        
        return np.std(values)
    
    def get_min_max(self, name: str, window: Optional[int] = None) -> Optional[tuple]:
        """Get min and max values of a metric over window."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        
        values = list(self.metrics[name])
        if window is not None:
            values = values[-window:]
        
        return (np.min(values), np.max(values))
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """
        Get statistical summary of a metric.
        
        Returns:
            Dictionary with mean, std, min, max, current
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {}
        
        values = list(self.metrics[name])
        
        return {
            'current': values[-1],
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get summaries for all tracked metrics."""
        return {name: self.get_summary(name) for name in self.metrics.keys()}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.global_metrics.clear()
        self.step = 0
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        data = {
            'step': self.step,
            'window_size': self.window_size,
            'metrics': {name: list(values) for name, values in self.metrics.items()},
            'global_metrics': dict(self.global_metrics),
            'summaries': self.get_all_summaries()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.step = data.get('step', 0)
        self.window_size = data.get('window_size', 100)
        
        # Restore metrics
        for name, values in data.get('metrics', {}).items():
            self.metrics[name] = deque(values, maxlen=self.window_size)
        
        # Restore global metrics
        for name, values in data.get('global_metrics', {}).items():
            self.global_metrics[name] = values


def compute_metrics(
    predictions: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    prefix: str = ''
) -> Dict[str, float]:
    """
    Compute common metrics for predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets (optional)
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics[f'{prefix}mean'] = predictions.mean().item()
    metrics[f'{prefix}std'] = predictions.std().item()
    metrics[f'{prefix}min'] = predictions.min().item()
    metrics[f'{prefix}max'] = predictions.max().item()
    
    # Norm metrics
    metrics[f'{prefix}l1_norm'] = predictions.abs().mean().item()
    metrics[f'{prefix}l2_norm'] = predictions.norm(p=2).item()
    
    # If targets provided, compute error metrics
    if targets is not None:
        error = predictions - targets
        metrics[f'{prefix}mse'] = (error ** 2).mean().item()
        metrics[f'{prefix}mae'] = error.abs().mean().item()
        metrics[f'{prefix}rmse'] = torch.sqrt((error ** 2).mean()).item()
        
        # Correlation if both have variance
        if predictions.std() > 1e-6 and targets.std() > 1e-6:
            pred_norm = (predictions - predictions.mean()) / predictions.std()
            target_norm = (targets - targets.mean()) / targets.std()
            correlation = (pred_norm * target_norm).mean().item()
            metrics[f'{prefix}correlation'] = correlation
    
    return metrics


def compute_causal_metrics(causal_output: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute metrics specific to causal reasoning module.
    
    Args:
        causal_output: Output from CausalReasoningModule
        
    Returns:
        Dictionary of causal metrics
    """
    metrics = {}
    
    if 'causal_graph' in causal_output:
        adj = causal_output['causal_graph']
        metrics['graph_sparsity'] = adj.mean().item()
        metrics['graph_density'] = (adj > 0.5).float().mean().item()
        metrics['max_edge_weight'] = adj.max().item()
        metrics['min_edge_weight'] = adj.min().item()
    
    if 'causal_effects' in causal_output:
        effects = causal_output['causal_effects']
        metrics['mean_causal_effect'] = effects.mean().item()
        metrics['max_causal_effect'] = effects.abs().max().item()
    
    if 'variables' in causal_output:
        variables = causal_output['variables']
        metrics['variable_mean_norm'] = variables.norm(dim=-1).mean().item()
        metrics['variable_std'] = variables.std().item()
    
    return metrics


def compute_intrinsic_metrics(intrinsic_signals: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute metrics for intrinsic motivation signals.
    
    Args:
        intrinsic_signals: Output from IntrinsicSignalSynthesizer
        
    Returns:
        Dictionary of intrinsic metrics
    """
    metrics = {}
    
    for key, value in intrinsic_signals.items():
        if isinstance(value, torch.Tensor):
            metrics[f'intrinsic_{key}_mean'] = value.mean().item()
            metrics[f'intrinsic_{key}_std'] = value.std().item()
    
    return metrics
