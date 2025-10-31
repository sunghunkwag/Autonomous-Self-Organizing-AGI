# Autonomous Self-Organizing AI System (Upgraded & Production-Ready) 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![No Rewards](https://img.shields.io/badge/Paradigm-Reward--Free-red.svg)](#)
[![Self-Organizing](https://img.shields.io/badge/Behavior-Autonomous-green.svg)](#)

A fully integrated, reward-free AGI system featuring intrinsic motivation, high-order meta-cognition, GNN-based causal reasoning, and multi-objective decision making. This upgraded version is now production-ready with comprehensive utilities for logging, checkpointing, metrics tracking, and visualization.

---

## 🚀 What’s New (Upgraded Engine)

This major upgrade focuses on robustness, usability, and observability, making the ASAGI system more practical for real-world experiments.

- **Production-Ready Utilities (`asagi/utils`):**
  - **Checkpoint Manager:** Automatically save and load model states, with versioning and metadata.
  - **Advanced Logger:** Structured, colored console logging and persistent file-based logs.
  - **Metrics Tracker:** Track, compute, and save dozens of system metrics over time.
  - **Visualizer:** Automatically generate plots for causal graphs, intrinsic motivation, and system performance.
- **Enhanced Demo (`examples/enhanced_demo.py`):** A new comprehensive script that showcases all new utilities in action.
- **Robustness & Validation:** Core modules have been updated with input validation, error handling, and more extensive type hints.
- **Comprehensive Test Suite (`tests/`):** A full suite of unit and integration tests to ensure system stability and correctness.

---

## 🏗️ System Architecture

The core architecture remains reward-free and self-organizing:

```
Meta-Cognition (goals, self-reflection)
        ↕
Intrinsic Signal Synthesizer (dissonance, compression gain, uncertainty, novelty)
        ↕
Pareto Navigator (loss-free multi-objective decisions)
        ↕
Operational Layer
  - World Model (Residual + Transformer Mixer + CPC)
  - Consistency Learner (EMA Teacher Alignment)
  - Causal Reasoning (GNN): structure, do(), counterfactuals, effects
        ↕
System Utilities (Logging, Checkpoints, Metrics, Visualization)
```

- **Reward-free:** No external reward functions, task losses, or policy objectives.
- **Emergent Goals:** Goals emerge from intrinsic signals and meta-cognitive analysis.
- **Observable:** All internal states and metrics can be logged, saved, and visualized.

---

## 🚦 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/Autonomous-Self-Organizing-AGI.git
cd Autonomous-Self-Organizing-AGI

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Enhanced Demonstration

Run the new, comprehensive demo to see all features in action. This is the recommended way to start.

```bash
python examples/enhanced_demo.py
```

This demo will:
- Initialize the full ASAGI system.
- Run for a number of steps, showing live, colored logging.
- Automatically save logs to the `/logs` directory.
- Automatically save performance metrics to the `/metrics` directory.
- Automatically generate and save visualizations to the `/visualizations` directory.
- Automatically save a final model checkpoint to the `/checkpoints` directory.

### Running Tests

To ensure all components are working correctly, run the test suite:

```bash
python tests/test_system.py
```

---

## 📁 Project Structure (Upgraded)

```
asagi/
  core/                      # Core system logic (unchanged)
    autonomous_system.py
    meta_cognition.py
    causal_reasoning.py
  intrinsic/                 # Intrinsic motivation (unchanged)
    signal_synthesizer.py
  meta_learning/             # Multi-objective learning (unchanged)
    pareto_navigator.py
  operational/               # World model and consistency (unchanged)
    world_model.py
    consistency_learner.py
  utils/                     # Production-ready utilities (NEW)
    checkpoint.py
    logging.py
    metrics.py
    visualization.py
    validation.py
examples/
  enhanced_demo.py           # Main demo with all features (NEW)
  basic_autonomous_operation.py
  causal_discovery_demo.py
tests/
  test_system.py             # Comprehensive test suite (NEW)
checkpoints/                 # Saved model checkpoints (NEW)
logs/                        # Log files (NEW)
metrics/                     # Saved metrics data (NEW)
visualizations/              # Saved plots and graphs (NEW)
```

---

## 📊 System Utilities & Observability

The upgraded system includes a suite of utilities for better analysis and robustness.

- **Logging:** All system outputs are logged to both the console (with colors) and a timestamped file in the `/logs` directory.
- **Metrics:** The `MetricsTracker` captures dozens of metrics (e.g., system coherence, autonomy level, causal graph sparsity) and saves them as a JSON file in the `/metrics` directory.
- **Visualization:** The `Visualizer` automatically generates PNG plots for key system dynamics, including intrinsic signals and causal graphs, saving them in the `/visualizations` directory.
- **Checkpoints:** The `CheckpointManager` saves the full model and optimizer state, allowing you to resume experiments or deploy a trained model. Checkpoints are stored in the `/checkpoints` directory.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

This project stands on the shoulders of giants in the information theory, causal machine learning, meta-learning, and self-supervised learning communities. It embraces a future of reward-free, self-organizing intelligence guided by intrinsic principles.
