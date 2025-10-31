"""
Checkpoint Management System
============================
Handles saving and loading of model checkpoints with metadata.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic versioning and metadata."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
        
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer state
            epoch: Current epoch number
            metrics: Performance metrics
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'timestamp': timestamp,
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Update history
            self.checkpoint_history.append({
                'path': str(checkpoint_path),
                'epoch': epoch,
                'timestamp': timestamp,
                'metrics': metrics
            })
            
            # Save metadata separately for easy access
            metadata_path = checkpoint_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'timestamp': timestamp,
                    'metrics': metrics,
                    'metadata': metadata
                }, f, indent=2)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load checkpoint on
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model state loaded from {checkpoint_path}")
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded")
            
            return {
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'metadata': checkpoint.get('metadata', {}),
                'timestamp': checkpoint.get('timestamp', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        return checkpoints[-1] if checkpoints else None
    
    def get_best_checkpoint(self, metric: str = 'loss', mode: str = 'min') -> Optional[Path]:
        """
        Get the checkpoint with the best metric value.
        
        Args:
            metric: Metric name to compare
            mode: 'min' or 'max' for optimization direction
            
        Returns:
            Path to best checkpoint or None
        """
        if not self.checkpoint_history:
            return None
        
        valid_checkpoints = [
            cp for cp in self.checkpoint_history 
            if cp.get('metrics') and metric in cp['metrics']
        ]
        
        if not valid_checkpoints:
            return None
        
        if mode == 'min':
            best = min(valid_checkpoints, key=lambda x: x['metrics'][metric])
        else:
            best = max(valid_checkpoints, key=lambda x: x['metrics'][metric])
        
        return Path(best['path'])
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max_checkpoints limit."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                try:
                    checkpoint.unlink()
                    # Also remove metadata file
                    metadata_file = checkpoint.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs
) -> None:
    """
    Simple checkpoint saving function.
    
    Args:
        model: Model to save
        path: Path to save checkpoint
        optimizer: Optional optimizer
        **kwargs: Additional data to save
    """
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    
    if optimizer is not None:
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint_data, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Simple checkpoint loading function.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer
        device: Device to load on
        
    Returns:
        Checkpoint data dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {path}")
    return checkpoint
