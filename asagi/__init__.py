"""
ASAGI - Autonomous Self-Organizing AGI
=====================================

Package init that wires together the core autonomous AGI components.
This system is reward-free, goal-emergent, and meta-cognitively driven.
"""

from .core.autonomous_system import (
    AutonomousSelfOrganizingAGI,
    ASAGIConfig,
    create_autonomous_agi,
)

from .core.meta_cognition import (
    MetaCognitiveController,
    create_meta_cognitive_system,
    analyze_meta_cognitive_patterns,
)

from .intrinsic.signal_synthesizer import (
    IntrinsicSignalSynthesizer,
    create_intrinsic_synthesizer,
    analyze_motivation_patterns,
)

from .meta_learning.pareto_navigator import (
    ParetoNavigator,
    create_pareto_navigator,
    setup_computational_constraints,
    analyze_multi_objective_performance,
)

__all__ = [
    # Main system
    "AutonomousSelfOrganizingAGI",
    "ASAGIConfig",
    "create_autonomous_agi",
    # Meta-cognition
    "MetaCognitiveController",
    "create_meta_cognitive_system",
    "analyze_meta_cognitive_patterns",
    # Intrinsic signals
    "IntrinsicSignalSynthesizer",
    "create_intrinsic_synthesizer",
    "analyze_motivation_patterns",
    # Pareto navigation
    "ParetoNavigator",
    "create_pareto_navigator",
    "setup_computational_constraints",
    "analyze_multi_objective_performance",
]
