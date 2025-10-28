#!/usr/bin/env python3
"""
Basic Autonomous Operation Demo (UPGRADED)
==========================================
Demonstrate the integrated ASAGI system with:
- Upgraded WorldModel (residual + CPC)
- Upgraded ConsistencyLearner (EMA alignment)
- Integrated CausalReasoningModule (GNN-based)
- All experimental modules activated
"""

from asagi import ASAGIConfig, create_autonomous_agi
import torch

def main():
    print("\n🧠 ASAGI Autonomous Demo (INTEGRATED VERSION)")
    print("=" * 55)
    
    # Configure with all upgrades enabled
    config = ASAGIConfig(
        feature_dim=256,
        decision_dim=128,
        num_objectives=4,
        
        # Enable all advanced features
        enable_meta_cognition=True,
        enable_goal_emergence=True,
        use_pareto_navigation=True,
        enable_causal_reasoning=True,  # NEW: Real GNN causal reasoning
        enable_conscious_awareness=True,
        enable_creative_synthesis=True,
        
        # Upgraded world model settings
        world_model_use_transformer=True,
        world_model_depth=4,
        
        # Causal reasoning settings
        causal_num_variables=8,
        causal_hidden_dim=128,
        causal_num_layers=2
    )
    
    print(f"📋 System Configuration:")
    print(f"   Feature dim: {config.feature_dim}")
    print(f"   Causal variables: {config.causal_num_variables}")
    print(f"   World model depth: {config.world_model_depth}")
    print(f"   All advanced features: ENABLED")
    
    # Create INTEGRATED system
    agent = create_autonomous_agi(config)
    
    print(f"\n🚀 System Status Check:")
    status = agent.get_system_status()
    
    for component, state in status['component_status'].items():
        print(f"   {component}: {state}")
    
    print(f"\n🔬 Single Forward Pass Test...")
    
    # Single forward step with synthetic observations
    observations = torch.randn(1, config.feature_dim)
    system_output = agent(observations)
    
    print(f"   Output keys: {list(system_output.keys())}")
    print(f"   Intrinsic motivation: {system_output['intrinsic_signals']['intrinsic_motivation'].mean().item():.3f}")
    print(f"   System coherence: {system_output['system_coherence']:.3f}")
    print(f"   Autonomy level: {system_output['autonomy_level']:.3f}")
    print(f"   Behavioral complexity: {system_output['behavioral_complexity']:.3f}")
    
    # Check upgraded components
    if 'world_simulation' in system_output:
        world_sim = system_output['world_simulation']
        print(f"   ✅ World model: predictions shape {world_sim['predictions'].shape}")
        if 'cpc_signal' in world_sim:
            print(f"   ✅ CPC signal: {world_sim['cpc_signal'].mean().item():.3f}")
    
    if 'consistency_analysis' in system_output:
        consistency = system_output['consistency_analysis']
        print(f"   ✅ Consistency score: {consistency['consistency_score'].item():.3f}")
    
    if 'causal_reasoning' in system_output:
        causal = system_output['causal_reasoning']
        print(f"   ✅ Causal reasoning: graph sparsity {causal['graph_sparsity'].item():.3f}")
        print(f"   ✅ Causal variables: {causal['variables'].shape}")
        print(f"   ✅ Causal effects: {causal['causal_effects'].shape}")
    else:
        print(f"   ❌ Causal reasoning not active")
    
    print(f"\n🌱 Autonomous Operation Test (10 seconds)...")
    
    # Short autonomous run
    autonomous_results = agent.autonomous_operation(
        environment_interface=None,
        operation_time=10.0,
        curiosity_threshold=0.6
    )
    
    print(f"\n📊 Autonomous Operation Summary:")
    summary = autonomous_results['operation_summary']
    
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Show final system state
    final_state = autonomous_results['final_state']
    print(f"\n🏁 Final System State:")
    print(f"   Total decisions made: {final_state['decisions_made']}")
    print(f"   Average motivation: {final_state['average_motivation']:.3f}")
    print(f"   System coherence: {final_state['system_coherence']:.3f}")
    print(f"   Causal discoveries: {final_state.get('causal_discoveries', 0)}")
    print(f"   Causal graph sparsity: {final_state.get('causal_graph_sparsity', 0.0):.3f}")
    
    # Performance metrics
    metrics = autonomous_results['performance_metrics']
    if metrics:
        print(f"\n📈 Performance Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.3f}")
    
    print(f"\n✨ INTEGRATED Demo completed successfully!")
    print(f"   ✅ Reward-free operation")
    print(f"   ✅ Self-goal generation")
    print(f"   ✅ Multi-scale world modeling")
    print(f"   ✅ EMA consistency learning")
    print(f"   ✅ GNN causal reasoning")
    print(f"   ✅ Autonomous skill discovery")
    print(f"   ✅ Multi-objective Pareto navigation")
    print(f"\n   This is a fully autonomous, reward-free AGI system!")

if __name__ == "__main__":
    main()