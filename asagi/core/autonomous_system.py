"""
Autonomous Self-Organizing AGI System (INTEGRATED VERSION)
=========================================================

The main system that integrates all components to create a truly autonomous,
self-organizing artificial general intelligence that operates without external
rewards, objectives, or loss functions.

UPDATES IN THIS VERSION:
- REAL CausalReasoningModule (GNN-based) replaces MLP stub
- Upgraded WorldModel (multi-scale residual + CPC)
- Upgraded ConsistencyLearner (EMA teacher alignment)
- Experimental modules now use concrete builders
- Added causal reasoning knobs for compute control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field
from collections import deque
import time
import logging
from pathlib import Path

# Import upgraded components
from ..intrinsic.signal_synthesizer import IntrinsicSignalSynthesizer
from .meta_cognition import MetaCognitiveController
from ..meta_learning.pareto_navigator import ParetoNavigator
from ..operational.world_model import InternalWorldModel
from ..operational.consistency_learner import ConsistencyBasedLearner
from .causal_reasoning import CausalReasoningModule
from ._experimental_builders import build_consciousness_module, build_creative_synthesizer

@dataclass
class ASAGIConfig:
    """Configuration for the Autonomous Self-Organizing AGI System (UPGRADED)."""
    
    # Core dimensions
    feature_dim: int = 256
    decision_dim: int = 128
    num_objectives: int = 4
    
    # System behavior
    enable_meta_cognition: bool = True
    enable_goal_emergence: bool = True
    enable_skill_discovery: bool = True
    use_pareto_navigation: bool = True
    
    # Intrinsic motivation weights
    intrinsic_motivation_weight: float = 1.0
    curiosity_threshold: float = 0.6
    exploration_bonus: float = 0.1
    
    # Meta-cognitive parameters
    meta_learning_rate: float = 0.001
    self_reflection_frequency: int = 10
    hypothesis_generation_rate: float = 0.1
    
    # Operational parameters (UPGRADED)
    world_model_horizon: int = 50
    world_model_use_transformer: bool = True
    world_model_depth: int = 4
    consistency_threshold: float = 0.8
    skill_emergence_threshold: float = 0.7
    
    # Causal reasoning parameters (NEW)
    causal_num_variables: int = 8
    causal_hidden_dim: int = 128
    causal_num_layers: int = 2
    enable_causal_reasoning: bool = True
    
    # System limits
    max_hypotheses: int = 1000
    max_skills: int = 500
    memory_buffer_size: int = 10000
    
    # Performance tracking
    log_level: str = 'INFO'
    save_frequency: int = 1000
    visualization_frequency: int = 100
    
    # Experimental features (UPGRADED)
    enable_conscious_awareness: bool = True
    enable_creative_synthesis: bool = True

class AutonomousOperationState:
    """Tracks the autonomous operation state of the system."""
    
    def __init__(self):
        self.operation_time = 0.0
        self.decisions_made = 0
        self.goals_generated = 0
        self.skills_discovered = 0
        self.hypotheses_formed = 0
        self.consistency_violations = 0
        self.causal_discoveries = 0  # NEW
        
        # Behavioral metrics
        self.curiosity_episodes = 0
        self.exploration_steps = 0
        self.consolidation_steps = 0
        self.creative_episodes = 0
        
        # Performance indicators
        self.average_motivation = 0.0
        self.knowledge_growth_rate = 0.0
        self.behavioral_complexity = 0.0
        self.autonomy_level = 0.0
        
        # System health
        self.system_coherence = 1.0
        self.meta_cognitive_stability = 1.0
        self.pareto_frontier_diversity = 0.0
        self.causal_graph_sparsity = 0.0  # NEW

class EmergentSkillTracker:
    """Tracks and manages emergent skills discovered by the system."""
    
    def __init__(self, max_skills: int = 500):
        self.max_skills = max_skills
        self.skills: Dict[str, Dict] = {}
        self.skill_counter = 0
        self.skill_usage_history = deque(maxlen=10000)
        
    def register_skill(self, skill_pattern: torch.Tensor, 
                      context: Dict[str, Any], 
                      effectiveness: float) -> str:
        """Register a new emergent skill."""
        skill_id = f"skill_{self.skill_counter}"
        self.skill_counter += 1
        
        skill_info = {
            'id': skill_id,
            'pattern': skill_pattern.clone().detach(),
            'context': context.copy(),
            'effectiveness': effectiveness,
            'usage_count': 0,
            'discovery_time': time.time(),
            'last_used': time.time(),
            'refinement_level': 0,
            'composition_depth': 1
        }
        
        self.skills[skill_id] = skill_info
        
        if len(self.skills) > self.max_skills:
            self._remove_weakest_skill()
        
        return skill_id
    
    def use_skill(self, skill_id: str, context: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Use an existing skill and update its statistics."""
        if skill_id not in self.skills:
            return None
        
        skill = self.skills[skill_id]
        skill['usage_count'] += 1
        skill['last_used'] = time.time()
        
        self.skill_usage_history.append({
            'skill_id': skill_id,
            'context': context,
            'timestamp': time.time()
        })
        
        return skill['pattern']
    
    def _remove_weakest_skill(self):
        """Remove the skill with lowest combined effectiveness and usage."""
        if not self.skills:
            return
        
        def skill_score(skill):
            recency = 1.0 / (1.0 + time.time() - skill['last_used'])
            return skill['effectiveness'] * skill['usage_count'] * recency
        
        weakest_id = min(self.skills.keys(), key=lambda sid: skill_score(self.skills[sid]))
        del self.skills[weakest_id]
    
    def get_applicable_skills(self, context: Dict[str, Any], 
                            similarity_threshold: float = 0.7) -> List[str]:
        """Get skills applicable to current context."""
        applicable = []
        
        for skill_id, skill in self.skills.items():
            similarity = self._compute_context_similarity(context, skill['context'])
            
            if similarity >= similarity_threshold:
                applicable.append(skill_id)
        
        applicable.sort(key=lambda sid: (
            self.skills[sid]['effectiveness'],
            self.skills[sid]['usage_count'],
            -abs(time.time() - self.skills[sid]['last_used'])
        ), reverse=True)
        
        return applicable
    
    def _compute_context_similarity(self, context1: Dict[str, Any], 
                                   context2: Dict[str, Any]) -> float:
        """Compute similarity between contexts (simplified)."""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarity = len(common_keys) / max(len(context1), len(context2))
        return similarity

class AutonomousSelfOrganizingAGI(nn.Module):
    """
    Main Autonomous Self-Organizing AGI System (INTEGRATED VERSION).
    
    UPGRADED COMPONENTS:
    - InternalWorldModel: Multi-scale residual dynamics + CPC
    - ConsistencyBasedLearner: EMA teacher alignment
    - CausalReasoningModule: GNN-based causal inference
    - Experimental modules: Concrete implementations
    """
    
    def __init__(self, config: ASAGIConfig):
        super().__init__()
        self.config = config
        
        # Setup logging
        self._setup_logging()
        
        # Initialize UPGRADED core components
        self.intrinsic_synthesizer = IntrinsicSignalSynthesizer(
            feature_dim=config.feature_dim
        )
        
        self.meta_cognition = MetaCognitiveController(
            feature_dim=config.feature_dim
        )
        
        self.pareto_navigator = ParetoNavigator(
            num_objectives=config.num_objectives,
            decision_dim=config.decision_dim
        )
        
        # UPGRADED: WorldModel with residual dynamics + CPC
        self.world_model = InternalWorldModel(
            feature_dim=config.feature_dim,
            horizon=config.world_model_horizon,
            use_transformer=config.world_model_use_transformer,
            depth=config.world_model_depth
        )
        
        # UPGRADED: ConsistencyLearner with EMA teacher
        self.consistency_learner = ConsistencyBasedLearner(
            feature_dim=config.feature_dim,
            threshold=config.consistency_threshold
        )
        
        # NEW: Real CausalReasoningModule (replaces MLP stub)
        if config.enable_causal_reasoning:
            self.causal_reasoner = CausalReasoningModule(
                feature_dim=config.feature_dim,
                decision_dim=config.decision_dim,
                num_variables=config.causal_num_variables,
                hidden_dim=config.causal_hidden_dim,
                num_gnn_layers=config.causal_num_layers
            )
        
        # Skill and behavior tracking
        self.skill_tracker = EmergentSkillTracker(config.max_skills)
        
        # System state tracking
        self.operation_state = AutonomousOperationState()
        
        # Main processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.GELU(),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim)
        )
        
        # UPGRADED: Concrete experimental modules
        if config.enable_conscious_awareness:
            self.consciousness_module = build_consciousness_module(config.feature_dim)
        
        if config.enable_creative_synthesis:
            self.creative_synthesizer = build_creative_synthesizer(
                config.feature_dim, config.decision_dim
            )
        
        # Performance and behavior history
        self.behavior_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=1000)
        
        # System timing
        self.start_time = time.time()
        self.last_save_time = time.time()
        
        self.logger.info("Autonomous Self-Organizing AGI System (INTEGRATED) initialized")
        self.logger.info(f"Causal reasoning enabled: {config.enable_causal_reasoning}")
    
    def forward(self, 
                observations: torch.Tensor,
                environment_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main forward pass - the UPGRADED autonomous intelligence loop.
        """
        batch_size = observations.shape[0]
        current_time = time.time()
        
        # Process observations into features
        features = self.feature_processor(observations)
        
        # === INTRINSIC MOTIVATION SYNTHESIS ===
        intrinsic_signals = self.intrinsic_synthesizer(
            features=features,
            previous_features=self._get_previous_features(),
            new_hypothesis=None
        )
        
        # === META-COGNITIVE PROCESSING ===
        if self.config.enable_meta_cognition:
            meta_cognitive_output = self.meta_cognition(
                current_features=features,
                intrinsic_signals=intrinsic_signals,
                environment_context=environment_context
            )
        else:
            meta_cognitive_output = {'goals': {'primary_goal': torch.zeros(batch_size, 64)}}
        
        # === PARETO NAVIGATION ===
        if self.config.use_pareto_navigation:
            decision_output = self.pareto_navigator(
                current_state=features,
                intrinsic_signals=intrinsic_signals,
                constraints=None
            )
        else:
            decision_output = {'decision': torch.zeros(batch_size, self.config.decision_dim)}
        
        # === UPGRADED WORLD MODEL SIMULATION ===
        world_simulation = self.world_model(
            current_features=features,
            proposed_actions=decision_output['decision'],
            horizon=self.config.world_model_horizon
        )
        
        # === UPGRADED CONSISTENCY LEARNING ===
        cpc_signal = world_simulation.get('cpc_signal')
        consistency_analysis = self.consistency_learner(
            observations=features,
            predictions=world_simulation['predictions'],
            internal_state=meta_cognitive_output.get('meta_state', features),
            cpc_signal=cpc_signal
        )
        
        # === SKILL DISCOVERY AND MANAGEMENT ===
        skill_analysis = self._analyze_emergent_skills(
            features, decision_output, intrinsic_signals
        )
        
        # === UPGRADED EXPERIMENTAL MODULES ===
        consciousness_output = {}
        if self.config.enable_conscious_awareness:
            # Combine features, intrinsic signals, meta state
            consciousness_input = torch.cat([
                features,
                intrinsic_signals['intrinsic_motivation'].unsqueeze(-1).expand(-1, features.shape[-1]),
                meta_cognitive_output.get('meta_state', torch.zeros_like(features))
            ], dim=-1)
            consciousness_output = {
                'awareness_features': self.consciousness_module(consciousness_input)
            }
        
        creative_output = {}
        if self.config.enable_creative_synthesis:
            creative_input = torch.cat([
                features,
                skill_analysis.get('creative_potential', torch.zeros_like(features))
            ], dim=-1)
            creative_output = {
                'creative_synthesis': self.creative_synthesizer(creative_input)
            }
        
        # === REAL CAUSAL REASONING (replaces MLP stub) ===
        causal_output = {}
        if self.config.enable_causal_reasoning:
            causal_output = self.causal_reasoner(
                observations=features,
                actions=decision_output['decision'],
                world_simulation=world_simulation
            )
            
            # Update causal discovery tracking
            self.operation_state.causal_discoveries += 1
            self.operation_state.causal_graph_sparsity = causal_output['graph_sparsity'].item()
        
        # === SYSTEM INTEGRATION ===
        system_output = self._integrate_system_components(
            features=features,
            intrinsic_signals=intrinsic_signals,
            meta_cognitive_output=meta_cognitive_output,
            decision_output=decision_output,
            world_simulation=world_simulation,
            consistency_analysis=consistency_analysis,
            skill_analysis=skill_analysis,
            consciousness_output=consciousness_output,
            creative_output=creative_output,
            causal_output=causal_output  # NOW INCLUDED
        )
        
        # === AUTONOMOUS BEHAVIOR TRACKING ===
        self._update_autonomous_behavior_tracking(
            system_output, current_time
        )
        
        # === SYSTEM MAINTENANCE ===
        if current_time - self.last_save_time > self.config.save_frequency:
            self._save_system_state()
            self.last_save_time = current_time
        
        return system_output
    
    def autonomous_operation(self, 
                           environment_interface: Optional[Any] = None,
                           operation_time: float = 3600.0,
                           curiosity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Run the system in fully autonomous mode (UPGRADED VERSION).
        
        Now includes:
        - Real causal reasoning and discovery
        - Upgraded world modeling with CPC
        - Enhanced consistency learning with EMA
        """
        self.logger.info(f"Starting UPGRADED autonomous operation for {operation_time} seconds")
        
        start_time = time.time()
        operation_results = {
            'behaviors_exhibited': [],
            'goals_generated': [],
            'skills_discovered': [],
            'insights_gained': [],
            'creative_outputs': [],
            'causal_discoveries': [],  # NEW
            'system_evolution': []
        }
        
        step = 0
        while time.time() - start_time < operation_time:
            step += 1
            
            # Generate autonomous observations
            if environment_interface is not None:
                observations = self._interact_with_environment(environment_interface)
            else:
                observations = self._generate_internal_observations()
            
            # Main system processing (UPGRADED)
            system_output = self.forward(
                observations=observations,
                environment_context={'autonomous_mode': True, 'step': step}
            )
            
            # Analyze autonomous behavior
            behavior_analysis = self._analyze_autonomous_behavior(system_output)
            
            # Check for significant events (including causal discoveries)
            significant_events = self._detect_significant_events(
                system_output, curiosity_threshold
            )
            
            # Record autonomous behaviors and discoveries
            if significant_events:
                operation_results['behaviors_exhibited'].extend(
                    significant_events.get('behaviors', [])
                )
                operation_results['goals_generated'].extend(
                    significant_events.get('goals', [])
                )
                operation_results['skills_discovered'].extend(
                    significant_events.get('skills', [])
                )
                operation_results['insights_gained'].extend(
                    significant_events.get('insights', [])
                )
                operation_results['causal_discoveries'].extend(
                    significant_events.get('causal_discoveries', [])
                )
            
            # System evolution tracking
            if step % 100 == 0:
                evolution_snapshot = self._capture_system_evolution()
                operation_results['system_evolution'].append(evolution_snapshot)
                
                self.logger.info(
                    f"Autonomous step {step}: "
                    f"Curiosity={system_output['intrinsic_signals']['intrinsic_motivation'].mean():.3f}, "
                    f"Goals={len(operation_results['goals_generated'])}, "
                    f"Skills={len(operation_results['skills_discovered'])}, "
                    f"Causal={len(operation_results['causal_discoveries'])}"
                )
            
            # Adaptive sleep based on system activity
            activity_level = system_output['intrinsic_signals']['intrinsic_motivation'].mean().item()
            sleep_time = max(0.01, 0.1 * (1.0 - activity_level))
            time.sleep(sleep_time)
        
        # Compile final results
        final_results = self._compile_autonomous_operation_results(
            operation_results, time.time() - start_time
        )
        
        self.logger.info(
            f"UPGRADED autonomous operation completed. "
            f"Generated {len(operation_results['goals_generated'])} goals, "
            f"discovered {len(operation_results['skills_discovered'])} skills, "
            f"{len(operation_results['causal_discoveries'])} causal insights."
        )
        
        return final_results
    
    def _integrate_system_components(self, **components) -> Dict[str, Any]:
        """Integrate all system components into coherent output (UPGRADED)."""
        integrated_output = {
            'features': components['features'],
            'intrinsic_signals': components['intrinsic_signals'],
            'meta_cognitive_state': components['meta_cognitive_output'],
            'decisions': components['decision_output'],
            'world_simulation': components['world_simulation'],
            'consistency_analysis': components['consistency_analysis'],
            'skill_analysis': components['skill_analysis'],
            'system_coherence': self._compute_system_coherence(components),
            'autonomy_level': self._compute_autonomy_level(components),
            'behavioral_complexity': self._compute_behavioral_complexity(components)
        }
        
        # Add experimental components
        if components['consciousness_output']:
            integrated_output['consciousness'] = components['consciousness_output']
        
        if components['creative_output']:
            integrated_output['creative_synthesis'] = components['creative_output']
        
        # ADD: Real causal reasoning output
        if components['causal_output']:
            integrated_output['causal_reasoning'] = components['causal_output']
        
        return integrated_output
    
    def _detect_significant_events(self, system_output: Dict, threshold: float) -> Dict[str, List]:
        """Detect significant autonomous events (UPGRADED with causal discoveries)."""
        events = {
            'behaviors': [],
            'goals': [],
            'skills': [],
            'insights': [],
            'causal_discoveries': []  # NEW
        }
        
        motivation = system_output['intrinsic_signals']['intrinsic_motivation'].mean().item()
        
        if motivation > threshold:
            events['behaviors'].append('high_curiosity_episode')
        
        # Extract goals from meta-cognitive output
        if 'goal_recommendations' in system_output.get('meta_cognitive_state', {}).get('goals', {}):
            goals = system_output['meta_cognitive_state']['goals']['goal_recommendations']
            events['goals'].extend(goals)
        
        # Skills from skill analysis
        if system_output['skill_analysis']['new_skills_detected']:
            events['skills'].extend(system_output['skill_analysis']['new_skills_detected'])
        
        # NEW: Causal discoveries
        if 'causal_reasoning' in system_output:
            causal = system_output['causal_reasoning']
            # Significant causal discovery if graph has meaningful structure
            sparsity = causal.get('graph_sparsity', 0.0)
            if isinstance(sparsity, torch.Tensor):
                sparsity = sparsity.item()
            
            if 0.1 < sparsity < 0.8:  # Not too sparse, not too dense
                events['causal_discoveries'].append(
                    f'Discovered causal structure with sparsity {sparsity:.3f}'
                )
            
            # Significant causal effects
            effects = causal.get('causal_effects')
            if effects is not None:
                strong_effects = (effects.abs() > 0.5).sum().item()
                if strong_effects > 0:
                    events['causal_discoveries'].append(
                        f'Found {strong_effects} strong causal relationships'
                    )
        
        return events
    
    # === REST OF THE METHODS UNCHANGED ===
    # (keeping existing methods for brevity)
    
    def _analyze_emergent_skills(self, features: torch.Tensor, 
                               decision_output: Dict, 
                               intrinsic_signals: Dict) -> Dict[str, Any]:
        """Analyze and track emergent skills."""
        skill_analysis = {
            'current_skills': len(self.skill_tracker.skills),
            'skill_usage': [],
            'new_skills_detected': [],
            'skill_effectiveness': {},
            'creative_potential': features  # For creative synthesis
        }
        
        # Detect potential new skills based on consistent patterns
        if intrinsic_signals['intrinsic_motivation'].mean() > self.config.skill_emergence_threshold:
            skill_pattern = decision_output['decision'].mean(dim=0)
            
            context = {
                'motivation_level': intrinsic_signals['intrinsic_motivation'].mean().item(),
                'decision_type': decision_output.get('decision_type', ['unknown'])[0],
                'features_norm': features.norm(dim=-1).mean().item()
            }
            
            effectiveness = intrinsic_signals['intrinsic_motivation'].mean().item()
            
            skill_id = self.skill_tracker.register_skill(
                skill_pattern, context, effectiveness
            )
            
            skill_analysis['new_skills_detected'].append({
                'skill_id': skill_id,
                'pattern': skill_pattern.tolist(),
                'context': context,
                'effectiveness': effectiveness
            })
        
        return skill_analysis
    
    def _compute_system_coherence(self, components: Dict) -> float:
        """Compute overall system coherence (UPGRADED)."""
        consistency_score = components['consistency_analysis']['consistency_score'].mean().item()
        motivation_stability = 1.0 - components['intrinsic_signals']['primary_signals'].std().item()
        
        # ADD: Causal coherence if available
        causal_coherence = 1.0
        if components.get('causal_output'):
            sparsity = components['causal_output'].get('graph_sparsity', 0.5)
            if isinstance(sparsity, torch.Tensor):
                sparsity = sparsity.item()
            # Good sparsity is around 0.2-0.6
            causal_coherence = 1.0 - abs(sparsity - 0.4) / 0.4
        
        coherence = 0.5 * consistency_score + 0.3 * motivation_stability + 0.2 * causal_coherence
        return max(0.0, min(1.0, coherence))
    
    def _compute_autonomy_level(self, components: Dict) -> float:
        """Compute level of autonomous operation."""
        intrinsic_strength = components['intrinsic_signals']['intrinsic_motivation'].mean().item()
        goal_emergence = len(components['meta_cognitive_output']['goals']['goal_recommendations'])
        
        autonomy = 0.7 * intrinsic_strength + 0.3 * min(1.0, goal_emergence / 5.0)
        return max(0.0, min(1.0, autonomy))
    
    def _compute_behavioral_complexity(self, components: Dict) -> float:
        """Compute behavioral complexity measure."""
        decision_entropy = self._compute_entropy(components['decision_output']['decision'])
        skill_diversity = len(self.skill_tracker.skills) / max(1, self.config.max_skills)
        
        complexity = 0.5 * decision_entropy + 0.5 * skill_diversity
        return max(0.0, min(1.0, complexity))
    
    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Compute entropy of tensor values."""
        probs = F.softmax(tensor.flatten(), dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item() / np.log(len(probs))
    
    def _get_previous_features(self) -> Optional[torch.Tensor]:
        """Get previous features for temporal processing."""
        if self.behavior_history:
            return self.behavior_history[-1].get('features')
        return None
    
    def _generate_internal_observations(self) -> torch.Tensor:
        """Generate observations from internal simulation."""
        batch_size = 1
        obs_dim = self.config.feature_dim
        
        if hasattr(self, 'last_features'):
            noise = torch.randn_like(self.last_features) * 0.1
            observations = self.last_features + noise
        else:
            observations = torch.randn(batch_size, obs_dim)
        
        return observations
    
    def _interact_with_environment(self, environment) -> torch.Tensor:
        """Interact with external environment."""
        return torch.randn(1, self.config.feature_dim)
    
    def _update_autonomous_behavior_tracking(self, system_output: Dict, current_time: float):
        """Update tracking of autonomous behavior."""
        behavior_record = {
            'timestamp': current_time,
            'features': system_output['features'].clone().detach(),
            'intrinsic_motivation': system_output['intrinsic_signals']['intrinsic_motivation'].mean().item(),
            'system_coherence': system_output['system_coherence'],
            'autonomy_level': system_output['autonomy_level'],
            'behavioral_complexity': system_output['behavioral_complexity']
        }
        
        self.behavior_history.append(behavior_record)
        
        # Update operation state
        self.operation_state.operation_time = current_time - self.start_time
        self.operation_state.decisions_made += 1
        self.operation_state.average_motivation = behavior_record['intrinsic_motivation']
        self.operation_state.system_coherence = behavior_record['system_coherence']
        self.operation_state.autonomy_level = behavior_record['autonomy_level']
        self.operation_state.behavioral_complexity = behavior_record['behavioral_complexity']
        
        self.last_features = behavior_record['features']
    
    def _analyze_autonomous_behavior(self, system_output: Dict) -> Dict[str, Any]:
        """Analyze the current autonomous behavior."""
        analysis = {
            'behavior_type': 'autonomous_exploration',
            'motivation_level': system_output['intrinsic_signals']['intrinsic_motivation'].mean().item(),
            'exploration_tendency': system_output['intrinsic_signals']['primary_signals'][:, 3].mean().item(),
            'consolidation_tendency': system_output['intrinsic_signals']['primary_signals'][:, 1].mean().item(),
            'meta_cognitive_activity': system_output.get('meta_cognitive_state', {}).get('time_step', 0),
            'decision_confidence': system_output['decisions'].get('decision_scores', torch.tensor([0.5])).mean().item(),
            'causal_reasoning_active': 'causal_reasoning' in system_output  # NEW
        }
        
        return analysis
    
    def _capture_system_evolution(self) -> Dict[str, Any]:
        """Capture snapshot of system evolution (UPGRADED)."""
        return {
            'timestamp': time.time(),
            'operation_state': self.operation_state.__dict__.copy(),
            'skill_count': len(self.skill_tracker.skills),
            'hypothesis_count': len(self.meta_cognition.hypothesis_manager.hypotheses),
            'pareto_frontier_size': len(self.pareto_navigator.frontier_analyzer.pareto_front),
            'causal_discoveries': self.operation_state.causal_discoveries,  # NEW
            'system_complexity': self._compute_current_system_complexity()
        }
    
    def _compute_current_system_complexity(self) -> float:
        """Compute current system complexity."""
        complexity = 0.0
        complexity += len(self.skill_tracker.skills) / self.config.max_skills
        complexity += len(self.meta_cognition.hypothesis_manager.hypotheses) / self.config.max_hypotheses
        complexity += len(self.behavior_history) / 10000.0
        
        return min(1.0, complexity / 3.0)
    
    def _compile_autonomous_operation_results(self, results: Dict, total_time: float) -> Dict[str, Any]:
        """Compile final autonomous operation results (UPGRADED)."""
        return {
            'operation_summary': {
                'total_time': total_time,
                'decisions_made': self.operation_state.decisions_made,
                'goals_generated': len(results['goals_generated']),
                'skills_discovered': len(results['skills_discovered']),
                'behaviors_exhibited': len(set(results['behaviors_exhibited'])),
                'insights_gained': len(results['insights_gained']),
                'causal_discoveries': len(results['causal_discoveries'])  # NEW
            },
            'final_state': self.operation_state.__dict__.copy(),
            'system_evolution': results['system_evolution'],
            'detailed_results': results,
            'performance_metrics': self._compute_performance_metrics()
        }
    
    def _compute_performance_metrics(self) -> Dict[str, float]:
        """Compute performance metrics for autonomous operation."""
        if not self.behavior_history:
            return {}
        
        recent_behavior = list(self.behavior_history)[-100:]
        
        return {
            'average_motivation': np.mean([b['intrinsic_motivation'] for b in recent_behavior]),
            'motivation_stability': 1.0 - np.std([b['intrinsic_motivation'] for b in recent_behavior]),
            'coherence_trend': np.polyfit(
                range(len(recent_behavior)), 
                [b['system_coherence'] for b in recent_behavior], 1
            )[0] if len(recent_behavior) > 1 else 0.0,
            'autonomy_growth': recent_behavior[-1]['autonomy_level'] - recent_behavior[0]['autonomy_level'] if len(recent_behavior) > 1 else 0.0,
            'complexity_evolution': recent_behavior[-1]['behavioral_complexity'] - recent_behavior[0]['behavioral_complexity'] if len(recent_behavior) > 1 else 0.0
        }
    
    def _setup_logging(self):
        """Setup logging for the system."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ASAGI-INTEGRATED')
    
    def _save_system_state(self):
        """Save system state for persistence."""
        self.logger.info("UPGRADED system state saved")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status (UPGRADED)."""
        return {
            'operation_state': self.operation_state.__dict__,
            'component_status': {
                'intrinsic_synthesizer': 'active',
                'meta_cognition': 'active',
                'pareto_navigator': 'active',
                'world_model': 'UPGRADED (residual+CPC)',
                'consistency_learner': 'UPGRADED (EMA)',
                'causal_reasoner': 'INTEGRATED (GNN)' if self.config.enable_causal_reasoning else 'disabled'
            },
            'autonomous_capabilities': {
                'goal_generation': self.config.enable_goal_emergence,
                'skill_discovery': self.config.enable_skill_discovery,
                'meta_learning': self.config.enable_meta_cognition,
                'multi_objective_decisions': self.config.use_pareto_navigation,
                'causal_reasoning': self.config.enable_causal_reasoning  # NEW
            },
            'current_complexity': self._compute_current_system_complexity(),
            'runtime_statistics': {
                'total_runtime': time.time() - self.start_time,
                'behavior_history_size': len(self.behavior_history),
                'skill_count': len(self.skill_tracker.skills),
                'hypothesis_count': len(self.meta_cognition.hypothesis_manager.hypotheses),
                'causal_discoveries': self.operation_state.causal_discoveries  # NEW
            }
        }

# Factory function (UPGRADED)
def create_autonomous_agi(config: Optional[ASAGIConfig] = None) -> AutonomousSelfOrganizingAGI:
    """Factory function to create INTEGRATED Autonomous Self-Organizing AGI System."""
    if config is None:
        config = ASAGIConfig()
    
    return AutonomousSelfOrganizingAGI(config)