"""
Visualization Tools
==================
Provides visualization utilities for ASAGI system analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import networkx as nx


class Visualizer:
    """Visualization tools for ASAGI system."""
    
    def __init__(self, save_dir: str = './visualizations'):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_causal_graph(
        self,
        adjacency: torch.Tensor,
        threshold: float = 0.3,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize causal graph structure.
        
        Args:
            adjacency: Adjacency matrix [N, N]
            threshold: Edge weight threshold for visualization
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy and apply threshold
        adj_np = adjacency.detach().cpu().numpy()
        if adj_np.ndim == 3:
            adj_np = adj_np[0]  # Take first batch
        
        # Create directed graph
        G = nx.DiGraph()
        n_nodes = adj_np.shape[0]
        
        # Add nodes
        for i in range(n_nodes):
            G.add_node(i, label=f'V{i}')
        
        # Add edges above threshold
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and adj_np[i, j] > threshold:
                    G.add_edge(i, j, weight=adj_np[i, j])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color='lightblue',
            node_size=1000, alpha=0.9, ax=ax
        )
        
        # Draw edges with width based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(
            G, pos, width=[w * 3 for w in weights],
            alpha=0.6, edge_color=weights,
            edge_cmap=plt.cm.Blues, arrows=True,
            arrowsize=20, ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, {i: f'V{i}' for i in range(n_nodes)},
            font_size=12, font_weight='bold', ax=ax
        )
        
        ax.set_title('Causal Graph Structure', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_adjacency_matrix(
        self,
        adjacency: torch.Tensor,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot adjacency matrix as heatmap.
        
        Args:
            adjacency: Adjacency matrix
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        adj_np = adjacency.detach().cpu().numpy()
        if adj_np.ndim == 3:
            adj_np = adj_np[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            adj_np, annot=True, fmt='.2f',
            cmap='Blues', square=True,
            cbar_kws={'label': 'Edge Weight'},
            ax=ax
        )
        
        ax.set_title('Causal Adjacency Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Target Variable', fontsize=12)
        ax.set_ylabel('Source Variable', fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_intrinsic_signals(
        self,
        signals: Dict[str, List[float]],
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot intrinsic motivation signals over time.
        
        Args:
            signals: Dictionary of signal name to values
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        signal_names = list(signals.keys())[:4]
        
        for idx, name in enumerate(signal_names):
            values = signals[name]
            axes[idx].plot(values, linewidth=2, alpha=0.8)
            axes[idx].set_title(name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Step', fontsize=10)
            axes[idx].set_ylabel('Value', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Intrinsic Motivation Signals', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_system_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot system-level metrics.
        
        Args:
            metrics: Dictionary of metric name to values
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, values) in enumerate(metrics.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            ax.plot(values, linewidth=2, alpha=0.8)
            ax.set_title(name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_xlabel('Step', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('System Performance Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = 'Training Curves'
) -> plt.Figure:
    """
    Plot training curves for multiple metrics.
    
    Args:
        metrics: Dictionary of metric name to values
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, values in metrics.items():
        ax.plot(values, label=name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
