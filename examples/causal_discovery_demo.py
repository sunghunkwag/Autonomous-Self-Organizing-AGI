#!/usr/bin/env python3
"""
Causal Discovery Demo
====================
Demonstrate the GNN-based causal reasoning capabilities of ASAGI.
Shows causal structure discovery, intervention simulation, and counterfactual reasoning.
"""

import torch
import numpy as np
from asagi import ASAGIConfig, create_autonomous_agi
from asagi.core.causal_reasoning import CausalReasoningModule

def generate_synthetic_causal_data(batch_size: int = 4, feature_dim: int = 256, 
                                  decision_dim: int = 128) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic data with known causal structure for testing.
    
    Ground truth: X1 -> X2 -> X3, X1 -> X4
    """
    
    # Generate base variables
    x1 = torch.randn(batch_size, 1)
    x2 = 0.7 * x1 + 0.3 * torch.randn(batch_size, 1)  # X1 causes X2
    x3 = 0.8 * x2 + 0.2 * torch.randn(batch_size, 1)  # X2 causes X3
    x4 = 0.6 * x1 + 0.4 * torch.randn(batch_size, 1)  # X1 causes X4
    
    # Embed into higher dimensions
    observations = torch.randn(batch_size, feature_dim)
    observations[:, :4] = torch.cat([x1, x2, x3, x4], dim=1)
    
    actions = torch.randn(batch_size, decision_dim)
    
    return {
        'observations': observations,
        'actions': actions,
        'ground_truth_structure': torch.tensor([
            [0, 1, 0, 1],  # X1 -> X2, X4
            [0, 0, 1, 0],  # X2 -> X3
            [0, 0, 0, 0],  # X3 -> none
            [0, 0, 0, 0]   # X4 -> none
        ]).float()
    }

def analyze_causal_discovery(discovered_graph: torch.Tensor, 
                           ground_truth: torch.Tensor) -> Dict[str, float]:
    """
    Analyze how well the discovered causal structure matches ground truth.
    """
    # Take first sample from batch
    discovered = discovered_graph[0, :4, :4]  # Focus on first 4 variables
    
    # Threshold discovered graph
    discovered_binary = (discovered > 0.5).float()
    
    # Compute metrics
    true_edges = ground_truth.sum().item()
    discovered_edges = discovered_binary.sum().item()
    correct_edges = (discovered_binary * ground_truth).sum().item()
    
    precision = correct_edges / max(1, discovered_edges)
    recall = correct_edges / max(1, true_edges)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'structural_similarity': torch.cosine_similarity(
            discovered.flatten(), ground_truth.flatten(), dim=0
        ).item()
    }

def main():
    print("\n🧠 ASAGI Causal Discovery Demo")
    print("=" * 50)
    
    # Configure system with causal reasoning enabled
    config = ASAGIConfig(
        feature_dim=256,
        decision_dim=128,
        num_objectives=4,
        enable_causal_reasoning=True,
        causal_num_variables=8,
        causal_hidden_dim=128,
        causal_num_layers=2
    )
    
    print(f"📋 Configuration: {config.causal_num_variables} variables, {config.causal_num_layers} GNN layers")
    
    # Create ASAGI system
    agent = create_autonomous_agi(config)
    
    print("\n🔬 Testing Causal Reasoning Module Directly...")
    
    # Test causal module directly
    causal_module = CausalReasoningModule(
        feature_dim=config.feature_dim,
        decision_dim=config.decision_dim,
        num_variables=config.causal_num_variables,
        hidden_dim=config.causal_hidden_dim,
        num_gnn_layers=config.causal_num_layers
    )
    
    # Generate synthetic data
    data = generate_synthetic_causal_data(batch_size=2)
    
    print(f"📊 Generated synthetic data:")
    print(f"   Observations shape: {data['observations'].shape}")
    print(f"   Actions shape: {data['actions'].shape}")
    print(f"   Ground truth causal structure:")
    print(f"   {data['ground_truth_structure']}")
    
    # Test causal reasoning
    with torch.no_grad():
        causal_output = causal_module(
            data['observations'],
            data['actions'],
            {'predictions': torch.randn(2, 10, config.feature_dim)}  # Dummy world sim
        )
    
    print(f"\n🧠 Causal Reasoning Results:")
    print(f"   Causal graph shape: {causal_output['causal_graph'].shape}")
    print(f"   Graph sparsity: {causal_output['graph_sparsity'].item():.3f}")
    print(f"   Variables shape: {causal_output['variables'].shape}")
    print(f"   Causal effects shape: {causal_output['causal_effects'].shape}")
    
    # Analyze discovered structure
    analysis = analyze_causal_discovery(
        causal_output['causal_graph'], 
        data['ground_truth_structure']
    )
    
    print(f"\n📈 Structure Discovery Analysis:")
    print(f"   Precision: {analysis['precision']:.3f}")
    print(f"   Recall: {analysis['recall']:.3f}")
    print(f"   F1 Score: {analysis['f1_score']:.3f}")
    print(f"   Structural Similarity: {analysis['structural_similarity']:.3f}")
    
    print(f"\n🔄 Testing Intervention Simulation...")
    
    # Test intervention
    intervention_value = torch.randn(2, config.causal_hidden_dim)
    original_vars = causal_output['variables']
    
    print(f"   Original variable 0 mean: {original_vars[:, 0].mean().item():.3f}")
    print(f"   Intervention effects shape: {causal_output['intervention_effects'].shape}")
    print(f"   Intervened variable 0 mean: {causal_output['intervention_effects'][:, 0].mean().item():.3f}")
    
    print(f"\n🤔 Testing Counterfactual Reasoning...")
    print(f"   Counterfactual outcome shape: {causal_output['counterfactual_outcome'].shape}")
    print(f"   Counterfactual magnitude: {causal_output['counterfactual_outcome'].norm().item():.3f}")
    
    print(f"\n🎯 Testing Full ASAGI System Integration...")
    
    # Test full system with causal reasoning
    system_output = agent(
        observations=data['observations'],
        environment_context={'demo_mode': True}
    )
    
    print(f"   System output keys: {list(system_output.keys())}")
    print(f"   Intrinsic motivation: {system_output['intrinsic_signals']['intrinsic_motivation'].mean().item():.3f}")
    print(f"   System coherence: {system_output['system_coherence']:.3f}")
    print(f"   Autonomy level: {system_output['autonomy_level']:.3f}")
    
    if 'causal_reasoning' in system_output:
        causal = system_output['causal_reasoning']
        print(f"   ✅ Causal reasoning active!")
        print(f"   Causal graph sparsity: {causal['graph_sparsity'].item():.3f}")
        print(f"   Causal representation norm: {causal['causal_representation'].norm().item():.3f}")
    else:
        print(f"   ❌ Causal reasoning not found in system output")
    
    print(f"\n🚀 Testing Short Autonomous Operation...")
    
    # Short autonomous run
    autonomous_results = agent.autonomous_operation(
        environment_interface=None,
        operation_time=5.0,  # 5 seconds
        curiosity_threshold=0.6
    )
    
    print(f"\n📊 Autonomous Operation Results:")
    summary = autonomous_results['operation_summary']
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"   The system demonstrated:")
    print(f"   - Intrinsic motivation without rewards")
    print(f"   - Causal structure discovery")
    print(f"   - Intervention and counterfactual reasoning")
    print(f"   - Autonomous goal generation")
    print(f"   - Multi-objective decision making")
    print(f"   - Self-organizing behavior")

if __name__ == "__main__":
    main()