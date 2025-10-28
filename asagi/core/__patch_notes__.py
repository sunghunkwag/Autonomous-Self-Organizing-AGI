"""
Wire upgraded operational and experimental modules into the autonomous system.
- Uses CPC signal in consistency learner
- Switch experimental builders to concrete modules
- Keep compute bounded and batched
"""

import torch
from ._experimental_builders import build_consciousness_module, build_creative_synthesizer
from ..operational.world_model import InternalWorldModel
from ..operational.consistency_learner import ConsistencyBasedLearner

# Patch points are explicit to keep diffs small.
