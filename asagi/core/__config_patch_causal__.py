"""
ASAGI Config Patch: knobs for causal compute & stability
"""

from dataclasses import dataclass

@dataclass
class _ASAGIConfigCausalPatch:
    causal_num_variables: int = 8
    causal_hidden_dim: int = 128
    causal_num_layers: int = 2
    enable_causal_reasoning: bool = True
