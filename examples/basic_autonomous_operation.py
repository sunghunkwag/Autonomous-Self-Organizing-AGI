#!/usr/bin/env python3
"""
Basic Autonomous Operation Demo
==============================
Run a short autonomous session of the reward-free, self-organizing AGI.
"""

from asagi import ASAGIConfig, create_autonomous_agi
import torch

def main():
    cfg = ASAGIConfig(feature_dim=256, decision_dim=128, num_objectives=4)
    agent = create_autonomous_agi(cfg)

    # One forward step
    obs = torch.randn(1, cfg.feature_dim)
    out = agent(obs)
    print("Intrinsic motivation:", float(out['intrinsic_signals']['intrinsic_motivation'].mean()))

    # Short autonomous run
    results = agent.autonomous_operation(environment_interface=None, operation_time=10, curiosity_threshold=0.6)
    print("Summary:", results['operation_summary'])

if __name__ == "__main__":
    main()
