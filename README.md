# Autonomous Self-Organizing AI System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![No Rewards](https://img.shields.io/badge/Paradigm-Reward--Free-red.svg)](#)
[![Self-Organizing](https://img.shields.io/badge/Behavior-Autonomous-green.svg)](#)

A reward-free AI system featuring intrinsic motivation, meta-cognition, GNN-based causal reasoning, and multi-objective decision making. Includes utilities for logging, checkpointing, metrics tracking, and visualization.

---

## Recent Updates

This version focuses on robustness, usability, and observability for research experiments.

- **System Utilities (`asagi/utils`):**
  - **Checkpoint Manager:** Save and load model states with versioning and metadata
  - **Logger:** Structured console logging and persistent file-based logs
  - **Metrics Tracker:** Track and save system metrics over time
  - **Visualizer:** Generate plots for causal graphs, intrinsic motivation, and performance
- **Enhanced Demo (`examples/enhanced_demo.py`):** Comprehensive script showcasing utilities
- **Robustness:** Core modules updated with input validation and error handling
- **Test Suite (`tests/`):** Unit and integration tests for system stability

---

## System Architecture

The core architecture is reward-free and self-organizing:

```
Meta-Cognition (goals, self-reflection)
        ↕
Intrinsic Signal Synthesizer (dissonance, compression gain, uncertainty, novelty)
        ↕
Pareto Navigator (multi-objective decisions)
        ↕
Operational Layer
  - World Model (Residual + Transformer Mixer + CPC)
  - Consistency Learner (EMA Teacher Alignment)
  - Causal Reasoning (GNN): structure, do(), counterfactuals, effects
        ↕
System Utilities (Logging, Checkpoints, Metrics, Visualization)
```

- **Reward-free:** No external reward functions, task losses, or policy objectives
- **Emergent Goals:** Goals emerge from intrinsic signals and meta-cognitive analysis
- **Observable:** Internal states and metrics can be logged, saved, and visualized

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/Autonomous-Self-Organizing-AI.git
cd Autonomous-Self-Organizing-AI

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Demo

Run the comprehensive demo to see features in action:

```bash
python examples/enhanced_demo.py
```

This demo will:
- Initialize the ASAGI system
- Run for a number of steps with live logging
- Save logs to the `/logs` directory
- Save performance metrics to the `/metrics` directory
- Generate and save visualizations to the `/visualizations` directory
- Save a final model checkpoint to the `/checkpoints` directory

### Running Tests

To ensure components are working correctly:

```bash
python tests/test_system.py
```

---

## Project Structure

```
asagi/
  core/                      # Core system logic
    autonomous_system.py
    meta_cognition.py
    causal_reasoning.py
  intrinsic/                 # Intrinsic motivation
    signal_synthesizer.py
  meta_learning/             # Multi-objective learning
    pareto_navigator.py
  operational/               # World model and consistency
    world_model.py
    consistency_learner.py
  utils/                     # System utilities
    checkpoint.py
    logging.py
    metrics.py
    visualization.py
    validation.py
examples/
  enhanced_demo.py           # Main demo
  basic_autonomous_operation.py
  causal_discovery_demo.py
tests/
  test_system.py             # Test suite
checkpoints/                 # Saved model checkpoints
logs/                        # Log files
metrics/                     # Saved metrics data
visualizations/              # Saved plots and graphs
```

---

## System Utilities

The system includes utilities for analysis and robustness:

- **Logging:** System outputs logged to console (with colors) and timestamped files in `/logs`
- **Metrics:** MetricsTracker captures metrics (system coherence, autonomy level, causal graph sparsity) and saves as JSON in `/metrics`
- **Visualization:** Visualizer generates PNG plots for system dynamics, intrinsic signals, and causal graphs in `/visualizations`
- **Checkpoints:** CheckpointManager saves full model and optimizer state for resuming experiments in `/checkpoints`

---

## License

MIT License

---

## Acknowledgments

This project builds on research in information theory, causal machine learning, meta-learning, and self-supervised learning communities. It explores reward-free, self-organizing intelligence guided by intrinsic principles.