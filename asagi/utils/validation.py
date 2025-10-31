"""
Input Validation Utilities
==========================
Provides input validation and error checking utilities.
"""

import torch
from typing import Optional, Tuple, Union


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dim: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> None:
    """
    Validate tensor properties.
    
    Args:
        tensor: Tensor to validate
        name: Name of tensor for error messages
        expected_shape: Expected shape (None values are wildcards)
        expected_dim: Expected number of dimensions
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether infinite values are allowed
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    # Check dimensions
    if expected_dim is not None and tensor.ndim != expected_dim:
        raise ValidationError(
            f"{name} expected {expected_dim} dimensions, got {tensor.ndim}"
        )
    
    # Check shape
    if expected_shape is not None:
        if len(expected_shape) != tensor.ndim:
            raise ValidationError(
                f"{name} shape mismatch: expected {len(expected_shape)} dims, "
                f"got {tensor.ndim}"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, tensor.shape)):
            if expected is not None and expected != actual:
                raise ValidationError(
                    f"{name} shape mismatch at dim {i}: expected {expected}, got {actual}"
                )
    
    # Check for NaN
    if not allow_nan and torch.isnan(tensor).any():
        raise ValidationError(f"{name} contains NaN values")
    
    # Check for infinity
    if not allow_inf and torch.isinf(tensor).any():
        raise ValidationError(f"{name} contains infinite values")
    
    # Check value range
    if min_value is not None:
        if tensor.min() < min_value:
            raise ValidationError(
                f"{name} contains values below minimum {min_value}: {tensor.min().item()}"
            )
    
    if max_value is not None:
        if tensor.max() > max_value:
            raise ValidationError(
                f"{name} contains values above maximum {max_value}: {tensor.max().item()}"
            )


def validate_config(config, required_fields: list) -> None:
    """
    Validate configuration object has required fields.
    
    Args:
        config: Configuration object
        required_fields: List of required field names
        
    Raises:
        ValidationError: If required fields are missing
    """
    missing = []
    for field in required_fields:
        if not hasattr(config, field):
            missing.append(field)
    
    if missing:
        raise ValidationError(
            f"Configuration missing required fields: {', '.join(missing)}"
        )


def validate_positive(value: Union[int, float], name: str) -> None:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to check
        name: Name for error message
        
    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_range(
    value: Union[int, float],
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> None:
    """
    Validate that a value is within a range.
    
    Args:
        value: Value to check
        name: Name for error message
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        
    Raises:
        ValidationError: If value is out of range
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")


def check_tensor_health(tensor: torch.Tensor, name: str) -> dict:
    """
    Check tensor health and return diagnostic information.
    
    Args:
        tensor: Tensor to check
        name: Name of tensor
        
    Returns:
        Dictionary with health metrics
    """
    health = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'min': tensor.min().item() if tensor.numel() > 0 else None,
        'max': tensor.max().item() if tensor.numel() > 0 else None,
        'mean': tensor.mean().item() if tensor.numel() > 0 else None,
        'std': tensor.std().item() if tensor.numel() > 0 else None,
    }
    
    # Add warnings
    warnings = []
    if health['has_nan']:
        warnings.append('Contains NaN values')
    if health['has_inf']:
        warnings.append('Contains infinite values')
    if health['std'] is not None and health['std'] < 1e-6:
        warnings.append('Very low variance (possible dead neurons)')
    if health['max'] is not None and abs(health['max']) > 1e6:
        warnings.append('Very large values (possible instability)')
    
    health['warnings'] = warnings
    health['is_healthy'] = len(warnings) == 0
    
    return health
