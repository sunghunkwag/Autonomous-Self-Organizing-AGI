#!/usr/bin/env python3
"""
Comprehensive System Tests
==========================
Tests for ASAGI system components with validation and error handling.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from asagi import ASAGIConfig, create_autonomous_agi
from asagi.operational.world_model import InternalWorldModel
from asagi.operational.consistency_learner import ConsistencyBasedLearner
from asagi.core.causal_reasoning import CausalReasoningModule


class TestWorldModel:
    """Test suite for InternalWorldModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = InternalWorldModel(feature_dim=256, horizon=50)
        assert model.feature_dim == 256
        assert model.horizon == 50
    
    def test_forward_pass(self):
        """Test forward pass with valid inputs."""
        model = InternalWorldModel(feature_dim=256, horizon=10)
        x = torch.randn(4, 256)
        actions = torch.randn(4, 128)
        
        output = model(x, actions)
        
        assert 'predictions' in output
        assert 'final_state' in output
        assert 'cpc_signal' in output
        assert output['predictions'].shape == (4, 10, 256)
        assert output['final_state'].shape == (4, 256)
    
    def test_input_validation(self):
        """Test input validation."""
        model = InternalWorldModel(feature_dim=256, horizon=10)
        
        # Wrong dimensions
        with pytest.raises(ValueError):
            x = torch.randn(4, 128)  # Wrong feature dim
            actions = torch.randn(4, 128)
            model(x, actions)
        
        # Wrong shape
        with pytest.raises(ValueError):
            x = torch.randn(4, 256, 10)  # 3D instead of 2D
            actions = torch.randn(4, 128)
            model(x, actions)
    
    def test_nan_handling(self):
        """Test NaN handling."""
        model = InternalWorldModel(feature_dim=256, horizon=10)
        x = torch.randn(4, 256)
        x[0, 0] = float('nan')
        actions = torch.randn(4, 128)
        
        # Should handle NaN gracefully
        output = model(x, actions)
        assert not torch.isnan(output['predictions']).any()


class TestConsistencyLearner:
    """Test suite for ConsistencyBasedLearner."""
    
    def test_initialization(self):
        """Test learner initialization."""
        learner = ConsistencyBasedLearner(feature_dim=256)
        assert learner.feature_dim == 256
    
    def test_forward_pass(self):
        """Test forward pass."""
        learner = ConsistencyBasedLearner(feature_dim=256)
        obs = torch.randn(4, 256)
        preds = torch.randn(4, 10, 256)
        state = torch.randn(4, 256)
        
        output = learner(obs, preds, state)
        
        assert 'alignment' in output
        assert 'adjustments' in output
        assert 'consistency_score' in output
    
    def test_ema_update(self):
        """Test EMA target update."""
        learner = ConsistencyBasedLearner(feature_dim=256)
        
        # Get initial target params
        initial_params = [p.clone() for p in learner.target.module.parameters()]
        
        # Forward pass should update target
        obs = torch.randn(4, 256)
        preds = torch.randn(4, 10, 256)
        state = torch.randn(4, 256)
        learner(obs, preds, state)
        
        # Check that target was updated
        current_params = list(learner.target.module.parameters())
        for init, curr in zip(initial_params, current_params):
            assert not torch.equal(init, curr)


class TestCausalReasoning:
    """Test suite for CausalReasoningModule."""
    
    def test_initialization(self):
        """Test module initialization."""
        module = CausalReasoningModule(
            feature_dim=256,
            decision_dim=128,
            num_variables=8
        )
        assert module.num_variables == 8
    
    def test_forward_pass(self):
        """Test forward pass."""
        module = CausalReasoningModule(
            feature_dim=256,
            decision_dim=128,
            num_variables=8
        )
        
        obs = torch.randn(4, 256)
        actions = torch.randn(4, 128)
        world_sim = {'predictions': torch.randn(4, 10, 256)}
        
        output = module(obs, actions, world_sim)
        
        assert 'causal_graph' in output
        assert 'variables' in output
        assert 'causal_effects' in output
        assert output['causal_graph'].shape == (4, 8, 8)
    
    def test_structure_discovery(self):
        """Test causal structure discovery."""
        module = CausalReasoningModule(
            feature_dim=256,
            decision_dim=128,
            num_variables=8
        )
        
        obs_seq = torch.randn(4, 10, 256)
        graph = module.discover_causal_structure(obs_seq)
        
        assert graph.shape == (4, 8, 8)
        assert (graph >= 0).all() and (graph <= 1).all()


class TestIntegratedSystem:
    """Test suite for integrated ASAGI system."""
    
    def test_system_creation(self):
        """Test system creation."""
        config = ASAGIConfig(
            feature_dim=256,
            decision_dim=128,
            enable_causal_reasoning=True
        )
        
        agent = create_autonomous_agi(config)
        assert agent is not None
    
    def test_single_forward_pass(self):
        """Test single forward pass through system."""
        config = ASAGIConfig(
            feature_dim=256,
            decision_dim=128,
            enable_causal_reasoning=True,
            world_model_horizon=10
        )
        
        agent = create_autonomous_agi(config)
        obs = torch.randn(2, 256)
        
        output = agent(obs)
        
        assert 'decisions' in output
        assert 'intrinsic_signals' in output
        assert 'system_coherence' in output
    
    def test_system_status(self):
        """Test system status reporting."""
        config = ASAGIConfig(feature_dim=256)
        agent = create_autonomous_agi(config)
        
        status = agent.get_system_status()
        
        assert 'component_status' in status
        assert 'operation_state' in status


def run_tests():
    """Run all tests."""
    print("Running ASAGI System Tests...")
    print("=" * 60)
    
    # World Model Tests
    print("\n[1/4] Testing World Model...")
    test_wm = TestWorldModel()
    test_wm.test_initialization()
    test_wm.test_forward_pass()
    test_wm.test_input_validation()
    test_wm.test_nan_handling()
    print("✓ World Model tests passed")
    
    # Consistency Learner Tests
    print("\n[2/4] Testing Consistency Learner...")
    test_cl = TestConsistencyLearner()
    test_cl.test_initialization()
    test_cl.test_forward_pass()
    test_cl.test_ema_update()
    print("✓ Consistency Learner tests passed")
    
    # Causal Reasoning Tests
    print("\n[3/4] Testing Causal Reasoning...")
    test_cr = TestCausalReasoning()
    test_cr.test_initialization()
    test_cr.test_forward_pass()
    test_cr.test_structure_discovery()
    print("✓ Causal Reasoning tests passed")
    
    # Integrated System Tests
    print("\n[4/4] Testing Integrated System...")
    test_sys = TestIntegratedSystem()
    test_sys.test_system_creation()
    test_sys.test_single_forward_pass()
    test_sys.test_system_status()
    print("✓ Integrated System tests passed")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_tests()
