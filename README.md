# Autonomous Self-Organizing AGI System (INTEGRATED) 🧠✨

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![No Rewards](https://img.shields.io/badge/Paradigm-Reward--Free-red.svg)](#)
[![Self-Organizing](https://img.shields.io/badge/Behavior-Autonomous-green.svg)](#)

A fully integrated, reward-free AGI system featuring intrinsic motivation, high-order meta-cognition, GNN-based causal reasoning, and multi-objective decision making. Now includes upgraded world modeling (residual dynamics + Transformer mixer + CPC), EMA-based consistency learning, and concrete experimental modules.

---

## 🚀 What’s New (Integrated Engine)
- GNN-based Causal Reasoning Module: structure discovery, interventions (do-operator), counterfactuals, causal effect estimation ([asagi/core/causal_reasoning.py](asagi/core/causal_reasoning.py))
- Upgraded World Model: multi-scale residual dynamics + optional Transformer mixer + CPC self-supervision ([asagi/operational/world_model.py](asagi/operational/world_model.py))
- Upgraded Consistency Learner: BYOL-style EMA teacher alignment, CPC-guided ([asagi/operational/consistency_learner.py](asagi/operational/consistency_learner.py))
- Autonomous System Integrated: causal module wired in, with compute knobs ([asagi/core/autonomous_system.py](asagi/core/autonomous_system.py))
- New Demos: integrated autonomous demo and causal discovery demo in [examples/](examples)

---

## 🏗️ Updated Architecture Overview
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
```

- Reward-free: no external reward functions, task losses, or policy objectives
- Goals emerge from intrinsic signals and meta-cognitive analysis
- Causal reasoning augments planning and analysis without scalar losses

---

## 🔬 Key Modules (Integrated)
- World Model: residual blocks, optional Transformer encoder, CPC head for self-supervised temporal consistency
- Consistency Learner: EMA teacher/student projections; alignment score used for internal coherence
- Causal Reasoning (GNN):
  - Variable extraction → attention-style edge scoring → directed adjacency
  - Vectorized message passing → causal propagation
  - Interventions do(X=x), counterfactual analysis, pairwise effect estimation
- Meta-Cognition: self-reflection, knowledge gaps, autonomous goal generation
- Pareto Navigator: preference learning + constraint satisfaction without scalarization

---

## ⚙️ Compute & Stability Knobs
Add these to ASAGIConfig to control cost and stability:
- causal_num_variables (default 8): increase carefully; cost ~ O(B·N²·H)
- causal_num_layers (default 2)
- causal_hidden_dim (default 128)
- enable_causal_reasoning (default True)
- world_model_use_transformer (default True)
- world_model_depth (default 4)

Recommended for single GPU/Colab:
- N=8, layers=2, hidden=128; temperature/sparsity for adjacency; consider top-k masking for larger N.

---

## 🚦 Quick Start

### Installation
```bash
git clone https://github.com/sunghunkwag/Autonomous-Self-Organizing-AGI.git
cd Autonomous-Self-Organizing-AGI
pip install -r requirements.txt
pip install -e .
```

### Integrated Autonomous Demo
```bash
python examples/basic_autonomous_operation.py
```
Shows: intrinsic motivation (no rewards), upgraded world model (CPC), EMA consistency, GNN causal reasoning, Pareto navigation, emergent skills.

### Causal Discovery Demo
```bash
python examples/causal_discovery_demo.py
```
Shows: structure discovery, intervention simulation, counterfactual reasoning, and simple structure matching analysis.

---

## 🧠 Reward-Free Principles (Unchanged)
- No external rewards or loss functions
- Goals emerge from intrinsic signals: predictive dissonance, compression gains, uncertainty reduction, novelty topology
- Multi-objective decision making without scalarization (Pareto frontier navigation)

---

## 📁 Project Structure (Updated)
```
asagi/
  core/
    autonomous_system.py         # Integrated system (GNN causal wired)
    meta_cognition.py            # High-order meta-cognition
    causal_reasoning.py          # GNN-based causal engine (NEW)
    _experimental_builders.py    # Concrete experimental module builders
  intrinsic/
    signal_synthesizer.py        # Intrinsic motivation signals
  meta_learning/
    pareto_navigator.py          # Loss-free multi-objective decisions
  operational/
    world_model.py               # Residual+Transformer+CPP world model (UPG)
    consistency_learner.py       # EMA teacher alignment (UPG)
examples/
  basic_autonomous_operation.py  # Integrated demo (UPG)
  causal_discovery_demo.py       # Causal reasoning demo (NEW)
```

---

## 📊 Metrics & Logging
- Autonomous Operation Summary: motivation, coherence, autonomy, causal discoveries
- Causal Graph Sparsity: [0,1] — monitor stability (ideal ~0.2–0.6)
- CPC Signal & Consistency Score: self-supervised learning health

---

## ⚠️ Notes on Scaling
- Compute grows with number of causal variables; keep N modest on small GPUs
- Use sparsity/temperature controls; add top‑k masking for larger graphs
- All components are batched and vectorized for GPU parallelism

---

## 📄 License
MIT License

---

## 🙏 Acknowledgments
Information theory, causal ML, meta-learning, and self-supervised learning communities. This project embraces reward-free, self-organizing intelligence guided by intrinsic principles.
