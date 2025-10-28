"""
High-Order Meta-Cognition System
===============================

The meta-cognitive system that enables true autonomy by:
1. Self-reflection: Understanding its own knowledge state and capabilities
2. Goal emergence: Generating goals from internal analysis rather than external specification  
3. Learning strategy adaptation: Deciding what to learn and how to learn it
4. Hypothesis management: Scientific reasoning and theory formation
5. Meta-learning enhancement: Learning how to learn more effectively

This system operates purely on intrinsic motivation without external objectives,
representing a fundamental shift from traditional AI approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from collections import deque
import math
from dataclasses import dataclass
from enum import Enum

class KnowledgeState(Enum):
    """Types of knowledge states the system can be in."""
    EXPLORING = "exploring"  # Actively gathering new information
    CONSOLIDATING = "consolidating"  # Integrating and organizing knowledge
    HYPOTHESIZING = "hypothesizing"  # Forming and testing theories
    REFLECTING = "reflecting"  # Self-analysis and meta-learning
    CREATING = "creating"  # Generating novel combinations/solutions

@dataclass
class Hypothesis:
    """Represents a hypothesis or theory the system has formed."""
    id: str
    description: str
    confidence: float
    evidence_count: int
    supporting_features: torch.Tensor
    prediction_accuracy: float
    creation_time: int
    last_updated: int
    parent_hypothesis: Optional[str] = None
    child_hypotheses: List[str] = None
    
    def __post_init__(self):
        if self.child_hypotheses is None:
            self.child_hypotheses = []

class SelfReflectionModule(nn.Module):
    """
    Self-reflection capabilities for meta-cognitive awareness.
    
    Maintains a model of the system's own knowledge, uncertainties,
    capabilities, and learning progress.
    """
    
    def __init__(self, feature_dim: int, knowledge_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.knowledge_dim = knowledge_dim
        
        # Knowledge state encoder
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(feature_dim, knowledge_dim * 2),
            nn.LayerNorm(knowledge_dim * 2),
            nn.GELU(),
            nn.Linear(knowledge_dim * 2, knowledge_dim),
            nn.LayerNorm(knowledge_dim),
            nn.GELU()
        )
        
        # Self-model: represents what the system knows about itself
        self.self_model = nn.Sequential(
            nn.Linear(knowledge_dim, knowledge_dim),
            nn.LayerNorm(knowledge_dim),
            nn.GELU(),
            nn.Linear(knowledge_dim, knowledge_dim // 2)
        )
        
        # Capability assessor
        self.capability_assessor = nn.Sequential(
            nn.Linear(knowledge_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 5)  # 5 capability dimensions
        )
        
        # Knowledge gap detector
        self.gap_detector = nn.MultiheadAttention(
            embed_dim=knowledge_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Learning progress tracker
        self.progress_tracker = nn.Sequential(
            nn.Linear(knowledge_dim * 2, knowledge_dim),
            nn.GELU(),
            nn.Linear(knowledge_dim, 1),
            nn.Sigmoid()
        )
        
        # Knowledge state classifier
        self.state_classifier = nn.Sequential(
            nn.Linear(knowledge_dim, 64),
            nn.GELU(),
            nn.Linear(64, len(KnowledgeState))
        )
        
        # Memory of past knowledge states
        self.register_buffer('knowledge_history', torch.zeros(100, knowledge_dim))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, current_features: torch.Tensor,
                intrinsic_signals: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform self-reflection on current knowledge state.
        
        Args:
            current_features: Current feature representations [B, feature_dim]
            intrinsic_signals: Current intrinsic motivation signals
            
        Returns:
            Dictionary containing self-reflection results
        """
        batch_size = current_features.shape[0]
        
        # Encode current knowledge state
        knowledge_state = self.knowledge_encoder(current_features)
        
        # Self-model: what does the system know about itself?
        self_representation = self.self_model(knowledge_state)
        
        # Assess current capabilities
        capabilities = self.capability_assessor(knowledge_state)
        capability_names = ['perception', 'reasoning', 'memory', 'creativity', 'adaptation']
        capability_dict = {name: capabilities[:, i] for i, name in enumerate(capability_names)}
        
        # Detect knowledge gaps using attention mechanism
        if self.history_ptr > 5:  # Need some history
            hist_size = min(self.history_ptr.item(), 100)
            historical_knowledge = self.knowledge_history[:hist_size].unsqueeze(0).expand(batch_size, -1, -1)
            
            gap_attention, gap_weights = self.gap_detector(
                knowledge_state.unsqueeze(1),  # Query: current knowledge
                historical_knowledge,          # Key: historical knowledge
                historical_knowledge           # Value: historical knowledge
            )
            
            # Knowledge gaps are areas with low attention weights
            knowledge_gaps = 1.0 - gap_weights.mean(dim=1)  # [B, hist_size]
            gap_magnitude = knowledge_gaps.max(dim=1)[0]  # [B]
        else:
            gap_magnitude = torch.zeros(batch_size, device=current_features.device)
            gap_attention = knowledge_state.unsqueeze(1)
        
        # Assess learning progress
        if self.history_ptr > 1:
            prev_knowledge = self.knowledge_history[max(0, self.history_ptr.item() - 1)]
            progress_input = torch.cat([
                knowledge_state, 
                prev_knowledge.unsqueeze(0).expand(batch_size, -1)
            ], dim=-1)
            learning_progress = self.progress_tracker(progress_input).squeeze(-1)
        else:
            learning_progress = torch.zeros(batch_size, device=current_features.device)
        
        # Classify current knowledge state
        state_logits = self.state_classifier(knowledge_state)
        knowledge_state_probs = F.softmax(state_logits, dim=-1)
        predicted_state = torch.argmax(state_logits, dim=-1)
        
        # Update knowledge history
        self._update_knowledge_history(knowledge_state)
        
        return {
            'knowledge_state': knowledge_state,
            'self_representation': self_representation,
            'capabilities': capability_dict,
            'knowledge_gaps': gap_magnitude,
            'learning_progress': learning_progress,
            'predicted_state': predicted_state,
            'state_probabilities': knowledge_state_probs,
            'gap_attention': gap_attention.squeeze(1)  # Remove sequence dimension
        }
    
    def _update_knowledge_history(self, knowledge_state: torch.Tensor):
        """Update the history of knowledge states."""
        # Take mean across batch for history
        state_summary = knowledge_state.mean(dim=0).detach()
        
        ptr = self.history_ptr.item()
        if ptr < 100:
            self.knowledge_history[ptr] = state_summary
            self.history_ptr[0] = ptr + 1
        else:
            # Circular buffer
            self.knowledge_history = torch.roll(self.knowledge_history, -1, dims=0)
            self.knowledge_history[-1] = state_summary

class HypothesisManager:
    """
    Manages scientific hypotheses generated by the system.
    
    Maintains a graph of hypotheses, tracks evidence, and manages
    the scientific reasoning process.
    """
    
    def __init__(self, max_hypotheses: int = 1000):
        self.max_hypotheses = max_hypotheses
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.hypothesis_counter = 0
        self.active_experiments: Set[str] = set()
        
    def create_hypothesis(self, description: str, supporting_features: torch.Tensor,
                         confidence: float = 0.5, parent_id: Optional[str] = None) -> str:
        """Create a new hypothesis."""
        hypothesis_id = f"hyp_{self.hypothesis_counter}"
        self.hypothesis_counter += 1
        
        hypothesis = Hypothesis(
            id=hypothesis_id,
            description=description,
            confidence=confidence,
            evidence_count=0,
            supporting_features=supporting_features.detach().clone(),
            prediction_accuracy=0.0,
            creation_time=self.hypothesis_counter,
            last_updated=self.hypothesis_counter,
            parent_hypothesis=parent_id
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        
        # Add to parent's children if parent exists
        if parent_id and parent_id in self.hypotheses:
            self.hypotheses[parent_id].child_hypotheses.append(hypothesis_id)
        
        # Remove oldest hypothesis if at capacity
        if len(self.hypotheses) > self.max_hypotheses:
            self._remove_weakest_hypothesis()
        
        return hypothesis_id
    
    def update_hypothesis(self, hypothesis_id: str, evidence: torch.Tensor,
                         prediction_accuracy: float):
        """Update a hypothesis with new evidence."""
        if hypothesis_id not in self.hypotheses:
            return
        
        hyp = self.hypotheses[hypothesis_id]
        hyp.evidence_count += 1
        hyp.last_updated = self.hypothesis_counter
        
        # Update prediction accuracy with exponential moving average
        alpha = 0.1
        hyp.prediction_accuracy = (1 - alpha) * hyp.prediction_accuracy + alpha * prediction_accuracy
        
        # Update confidence based on evidence and accuracy
        evidence_factor = min(1.0, hyp.evidence_count / 10.0)
        accuracy_factor = hyp.prediction_accuracy
        hyp.confidence = 0.5 * evidence_factor + 0.5 * accuracy_factor
    
    def get_best_hypotheses(self, n: int = 10) -> List[Hypothesis]:
        """Get the n best hypotheses by confidence and evidence."""
        hypothesis_list = list(self.hypotheses.values())
        
        # Score hypotheses by confidence * evidence_count * recency
        def score_hypothesis(hyp):
            recency = 1.0 / (1.0 + self.hypothesis_counter - hyp.last_updated)
            return hyp.confidence * math.log(1 + hyp.evidence_count) * recency
        
        hypothesis_list.sort(key=score_hypothesis, reverse=True)
        return hypothesis_list[:n]
    
    def generate_experiment(self, hypothesis_id: str) -> Dict[str, Any]:
        """Generate an experiment to test a hypothesis."""
        if hypothesis_id not in self.hypotheses:
            return {}
        
        hyp = self.hypotheses[hypothesis_id]
        
        experiment = {
            'hypothesis_id': hypothesis_id,
            'test_features': hyp.supporting_features,
            'expected_outcome': hyp.confidence,
            'experiment_type': self._determine_experiment_type(hyp)
        }
        
        self.active_experiments.add(hypothesis_id)
        return experiment
    
    def _determine_experiment_type(self, hypothesis: Hypothesis) -> str:
        """Determine what type of experiment would test this hypothesis."""
        if hypothesis.confidence < 0.3:
            return "exploratory"  # Low confidence, explore broadly
        elif hypothesis.confidence > 0.8:
            return "confirmatory"  # High confidence, seek confirmation
        else:
            return "comparative"  # Medium confidence, compare alternatives
    
    def _remove_weakest_hypothesis(self):
        """Remove the hypothesis with lowest score."""
        if not self.hypotheses:
            return
        
        def score_hypothesis(hyp):
            recency = 1.0 / (1.0 + self.hypothesis_counter - hyp.last_updated)
            return hyp.confidence * math.log(1 + hyp.evidence_count) * recency
        
        weakest_id = min(self.hypotheses.keys(), 
                        key=lambda h_id: score_hypothesis(self.hypotheses[h_id]))
        
        # Remove from parent's children list
        weakest = self.hypotheses[weakest_id]
        if weakest.parent_hypothesis and weakest.parent_hypothesis in self.hypotheses:
            parent = self.hypotheses[weakest.parent_hypothesis]
            if weakest_id in parent.child_hypotheses:
                parent.child_hypotheses.remove(weakest_id)
        
        del self.hypotheses[weakest_id]

class GoalEmergenceSystem(nn.Module):
    """
    System for autonomous goal generation based on internal analysis.
    
    Goals emerge from:
    1. Knowledge gaps identified through self-reflection
    2. High-potential areas identified by intrinsic signals
    3. Hypothesis testing needs
    4. Meta-learning opportunities
    """
    
    def __init__(self, feature_dim: int, goal_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.goal_dim = goal_dim
        
        # Goal generation network
        self.goal_generator = nn.Sequential(
            nn.Linear(feature_dim + 128, goal_dim * 2),  # +128 for intrinsic signals
            nn.LayerNorm(goal_dim * 2),
            nn.GELU(),
            nn.Linear(goal_dim * 2, goal_dim * 2),
            nn.LayerNorm(goal_dim * 2),
            nn.GELU(),
            nn.Linear(goal_dim * 2, goal_dim)
        )
        
        # Goal priority network
        self.priority_network = nn.Sequential(
            nn.Linear(goal_dim, goal_dim // 2),
            nn.GELU(),
            nn.Linear(goal_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Goal feasibility assessor
        self.feasibility_assessor = nn.Sequential(
            nn.Linear(goal_dim + feature_dim, goal_dim),
            nn.GELU(),
            nn.Linear(goal_dim, 1),
            nn.Sigmoid()
        )
        
        # Subgoal decomposer
        self.subgoal_decomposer = nn.Sequential(
            nn.Linear(goal_dim, goal_dim * 2),
            nn.GELU(),
            nn.Linear(goal_dim * 2, goal_dim * 4),  # 4 subgoals
            nn.GELU()
        )
        
        # Goal type classifier
        self.goal_classifier = nn.Sequential(
            nn.Linear(goal_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4)  # [explore, consolidate, hypothesize, create]
        )
        
    def forward(self, 
                current_features: torch.Tensor,
                self_reflection: Dict[str, torch.Tensor],
                intrinsic_signals: Dict[str, torch.Tensor],
                hypothesis_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate autonomous goals based on current state analysis.
        
        Args:
            current_features: Current feature representations [B, feature_dim]
            self_reflection: Self-reflection analysis results
            intrinsic_signals: Intrinsic motivation signals
            hypothesis_context: Context from hypothesis manager
            
        Returns:
            Dictionary containing generated goals and priorities
        """
        batch_size = current_features.shape[0]
        
        # Combine intrinsic signals into a compact representation
        signal_features = torch.stack([
            intrinsic_signals['primary_signals'][:, 0],  # Dissonance
            intrinsic_signals['primary_signals'][:, 1],  # Compression gain
            intrinsic_signals['primary_signals'][:, 2],  # Uncertainty reduction
            intrinsic_signals['primary_signals'][:, 3],  # Novelty
            self_reflection['knowledge_gaps'],
            self_reflection['learning_progress']
        ], dim=-1)  # [B, 6]
        
        # Pad to expected size (128)
        padding_size = 128 - signal_features.shape[-1]
        if padding_size > 0:
            padding = torch.zeros(batch_size, padding_size, device=current_features.device)
            signal_features = torch.cat([signal_features, padding], dim=-1)
        else:
            signal_features = signal_features[:, :128]
        
        # Generate primary goal
        goal_input = torch.cat([current_features, signal_features], dim=-1)
        primary_goal = self.goal_generator(goal_input)
        
        # Assess goal priority
        goal_priority = self.priority_network(primary_goal).squeeze(-1)
        
        # Assess feasibility
        feasibility_input = torch.cat([primary_goal, current_features], dim=-1)
        goal_feasibility = self.feasibility_assessor(feasibility_input).squeeze(-1)
        
        # Decompose into subgoals
        subgoals_flat = self.subgoal_decomposer(primary_goal)
        subgoals = subgoals_flat.view(batch_size, 4, self.goal_dim)
        
        # Classify goal type
        goal_type_logits = self.goal_classifier(primary_goal)
        goal_type_probs = F.softmax(goal_type_logits, dim=-1)
        
        # Generate specific goal recommendations based on analysis
        goal_recommendations = self._generate_specific_recommendations(
            self_reflection, intrinsic_signals
        )
        
        return {
            'primary_goal': primary_goal,
            'goal_priority': goal_priority,
            'goal_feasibility': goal_feasibility,
            'subgoals': subgoals,
            'goal_type_probabilities': goal_type_probs,
            'goal_recommendations': goal_recommendations,
            'overall_goal_score': goal_priority * goal_feasibility
        }
    
    def _generate_specific_recommendations(self,
                                         self_reflection: Dict[str, torch.Tensor],
                                         intrinsic_signals: Dict[str, torch.Tensor]) -> List[str]:
        """
        Generate human-readable goal recommendations.
        
        Args:
            self_reflection: Self-reflection results
            intrinsic_signals: Intrinsic motivation signals
            
        Returns:
            List of recommended goals
        """
        recommendations = []
        
        # Analyze signals to generate specific recommendations
        dissonance = intrinsic_signals['primary_signals'][:, 0].mean().item()
        compression_gain = intrinsic_signals['primary_signals'][:, 1].mean().item()
        uncertainty = intrinsic_signals['primary_signals'][:, 2].mean().item()
        novelty = intrinsic_signals['primary_signals'][:, 3].mean().item()
        
        knowledge_gaps = self_reflection['knowledge_gaps'].mean().item()
        learning_progress = self_reflection['learning_progress'].mean().item()
        
        # Generate recommendations based on dominant signals
        if dissonance > 0.7:
            recommendations.append("Investigate areas of high predictive error")
            recommendations.append("Refine models in domains with poor predictions")
        
        if compression_gain > 0.6:
            recommendations.append("Explore opportunities for knowledge compression")
            recommendations.append("Identify patterns that could simplify current models")
        
        if uncertainty > 0.6:
            recommendations.append("Focus learning on high-uncertainty regions")
            recommendations.append("Gather more evidence in ambiguous domains")
        
        if novelty > 0.7:
            recommendations.append("Investigate novel patterns and structures")
            recommendations.append("Explore uncharted regions of the state space")
        
        if knowledge_gaps > 0.5:
            recommendations.append("Fill identified knowledge gaps")
            recommendations.append("Connect isolated knowledge islands")
        
        if learning_progress < 0.3:
            recommendations.append("Adapt learning strategy for better progress")
            recommendations.append("Consider meta-learning approach changes")
        
        # If no specific recommendations, provide general ones
        if not recommendations:
            recommendations = [
                "Continue balanced exploration and consolidation",
                "Maintain curiosity-driven investigation",
                "Focus on knowledge integration and synthesis"
            ]
        
        return recommendations[:5]  # Limit to top 5 recommendations

class MetaCognitiveController(nn.Module):
    """
    Main meta-cognitive controller that coordinates all meta-cognitive functions.
    
    This is the "brain" that decides what the system should focus on learning,
    how it should adapt its learning strategies, and what goals to pursue.
    """
    
    def __init__(self, feature_dim: int, config: Optional[Dict] = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = config or {}
        
        # Core components
        self.self_reflection = SelfReflectionModule(feature_dim)
        self.goal_emergence = GoalEmergenceSystem(feature_dim)
        self.hypothesis_manager = HypothesisManager()
        
        # Meta-learning strategy adaptor
        self.strategy_adaptor = nn.Sequential(
            nn.Linear(feature_dim + 128, 256),  # +128 for meta-state
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64)  # Strategy parameters
        )
        
        # Attention mechanism for focus allocation
        self.attention_controller = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Meta-cognitive state tracker
        self.register_buffer('meta_state', torch.zeros(128))
        self.register_buffer('time_step', torch.zeros(1, dtype=torch.long))
        
        # Performance history
        self.performance_history = deque(maxlen=1000)
        
    def forward(self, 
                current_features: torch.Tensor,
                intrinsic_signals: Dict[str, torch.Tensor],
                environment_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main meta-cognitive processing loop.
        
        Args:
            current_features: Current feature representations [B, feature_dim]
            intrinsic_signals: Intrinsic motivation signals
            environment_context: Optional environmental context
            
        Returns:
            Dictionary containing meta-cognitive analysis and decisions
        """
        batch_size = current_features.shape[0]
        self.time_step += 1
        
        # Self-reflection: understand current knowledge state
        self_reflection_results = self.self_reflection(
            current_features, intrinsic_signals
        )
        
        # Generate hypothesis context if needed
        hypothesis_context = None
        if len(self.hypothesis_manager.hypotheses) > 0:
            best_hypotheses = self.hypothesis_manager.get_best_hypotheses(5)
            if best_hypotheses:
                hypothesis_features = [h.supporting_features for h in best_hypotheses]
                hypothesis_context = torch.stack(hypothesis_features).mean(dim=0)
        
        # Goal emergence: decide what to focus on next
        goal_results = self.goal_emergence(
            current_features,
            self_reflection_results,
            intrinsic_signals,
            hypothesis_context
        )
        
        # Adapt meta-learning strategy
        meta_input = torch.cat([
            current_features,
            self.meta_state.unsqueeze(0).expand(batch_size, -1)
        ], dim=-1)
        strategy_params = self.strategy_adaptor(meta_input)
        
        # Compute attention allocation
        attended_features, attention_weights = self.attention_controller(
            current_features.unsqueeze(1),  # Query
            current_features.unsqueeze(1),  # Key
            current_features.unsqueeze(1)   # Value
        )
        attended_features = attended_features.squeeze(1)
        
        # Update meta-state
        new_meta_state = self._update_meta_state(
            self_reflection_results,
            intrinsic_signals,
            goal_results
        )
        self.meta_state = new_meta_state.mean(dim=0)  # Average across batch
        
        # Generate specific learning decisions
        learning_decisions = self._generate_learning_decisions(
            self_reflection_results,
            goal_results,
            intrinsic_signals
        )
        
        # Update performance tracking
        self._update_performance_tracking(intrinsic_signals)
        
        return {
            'self_reflection': self_reflection_results,
            'goals': goal_results,
            'strategy_parameters': strategy_params,
            'attention_allocation': attention_weights.squeeze(1),
            'learning_decisions': learning_decisions,
            'meta_state': self.meta_state,
            'time_step': self.time_step.item(),
            'hypothesis_count': len(self.hypothesis_manager.hypotheses)
        }
    
    def generate_hypothesis(self, features: torch.Tensor, 
                          description: str, confidence: float = 0.5) -> str:
        """Generate a new hypothesis based on current observations."""
        return self.hypothesis_manager.create_hypothesis(
            description, features, confidence
        )
    
    def test_hypothesis(self, hypothesis_id: str, 
                       test_features: torch.Tensor, 
                       outcome: float) -> bool:
        """Test a hypothesis with new data."""
        self.hypothesis_manager.update_hypothesis(
            hypothesis_id, test_features, outcome
        )
        return True
    
    def get_current_focus(self) -> Dict[str, float]:
        """Get what the system is currently focusing on."""
        if len(self.performance_history) == 0:
            return {'exploration': 1.0}
        
        recent_performance = list(self.performance_history)[-10:]
        avg_motivation = np.mean([p['intrinsic_motivation'] for p in recent_performance])
        avg_uncertainty = np.mean([p['uncertainty'] for p in recent_performance])
        avg_novelty = np.mean([p['novelty'] for p in recent_performance])
        
        total = avg_motivation + avg_uncertainty + avg_novelty + 1e-8
        
        return {
            'exploration': avg_novelty / total,
            'exploitation': avg_motivation / total,
            'investigation': avg_uncertainty / total,
            'balanced': 1.0 / total
        }
    
    def _update_meta_state(self, 
                          self_reflection: Dict[str, torch.Tensor],
                          intrinsic_signals: Dict[str, torch.Tensor],
                          goal_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Update the meta-cognitive state based on current analysis."""
        batch_size = list(self_reflection.values())[0].shape[0]
        
        # Combine key meta-cognitive indicators
        meta_indicators = torch.stack([
            self_reflection['learning_progress'],
            self_reflection['knowledge_gaps'], 
            goal_results['goal_priority'],
            goal_results['goal_feasibility'],
            intrinsic_signals['intrinsic_motivation']
        ], dim=-1)  # [B, 5]
        
        # Expand to full meta-state dimension
        meta_expansion = nn.Linear(5, 128, device=meta_indicators.device)
        expanded_meta = meta_expansion(meta_indicators)
        
        return expanded_meta
    
    def _generate_learning_decisions(self,
                                   self_reflection: Dict[str, torch.Tensor],
                                   goal_results: Dict[str, torch.Tensor],
                                   intrinsic_signals: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate specific learning decisions based on meta-cognitive analysis."""
        decisions = {
            'focus_areas': [],
            'learning_rate_modulation': 1.0,
            'exploration_vs_exploitation': 0.5,
            'hypothesis_generation_rate': 0.1,
            'memory_consolidation_trigger': False,
            'strategy_change_needed': False
        }
        
        # Analyze current state to make decisions
        avg_uncertainty = intrinsic_signals['primary_signals'][:, 2].mean().item()
        avg_novelty = intrinsic_signals['primary_signals'][:, 3].mean().item()
        learning_progress = self_reflection['learning_progress'].mean().item()
        
        # Determine focus areas
        if avg_uncertainty > 0.7:
            decisions['focus_areas'].append('uncertainty_reduction')
            decisions['learning_rate_modulation'] = 1.2  # Learn faster in uncertain areas
        
        if avg_novelty > 0.6:
            decisions['focus_areas'].append('novelty_exploration')
            decisions['exploration_vs_exploitation'] = 0.8  # More exploration
        
        if learning_progress < 0.3:
            decisions['strategy_change_needed'] = True
            decisions['focus_areas'].append('meta_learning')
        
        # Decide on hypothesis generation
        if avg_uncertainty > 0.5 and avg_novelty > 0.5:
            decisions['hypothesis_generation_rate'] = 0.3  # Generate more hypotheses
        
        # Memory consolidation trigger
        if learning_progress > 0.7 and avg_uncertainty < 0.4:
            decisions['memory_consolidation_trigger'] = True
        
        return decisions
    
    def _update_performance_tracking(self, intrinsic_signals: Dict[str, torch.Tensor]):
        """Update performance history for meta-learning."""
        performance_snapshot = {
            'time_step': self.time_step.item(),
            'intrinsic_motivation': intrinsic_signals['intrinsic_motivation'].mean().item(),
            'uncertainty': intrinsic_signals['primary_signals'][:, 2].mean().item(),
            'novelty': intrinsic_signals['primary_signals'][:, 3].mean().item(),
            'dissonance': intrinsic_signals['primary_signals'][:, 0].mean().item()
        }
        
        self.performance_history.append(performance_snapshot)

# Utility functions
def create_meta_cognitive_system(feature_dim: int, config: Optional[Dict] = None) -> MetaCognitiveController:
    """Factory function to create meta-cognitive system."""
    return MetaCognitiveController(feature_dim, config)

def analyze_meta_cognitive_patterns(controller: MetaCognitiveController) -> Dict[str, Any]:
    """Analyze patterns in meta-cognitive behavior over time."""
    if not controller.performance_history:
        return {'status': 'insufficient_data'}
    
    history = list(controller.performance_history)
    
    # Extract time series
    motivation_series = [h['intrinsic_motivation'] for h in history]
    uncertainty_series = [h['uncertainty'] for h in history]
    novelty_series = [h['novelty'] for h in history]
    
    analysis = {
        'learning_stability': np.std(motivation_series),
        'curiosity_trend': np.polyfit(range(len(novelty_series)), novelty_series, 1)[0],
        'uncertainty_resolution': uncertainty_series[0] - uncertainty_series[-1] if len(uncertainty_series) > 1 else 0,
        'average_motivation': np.mean(motivation_series),
        'focus_areas': controller.get_current_focus(),
        'hypothesis_productivity': len(controller.hypothesis_manager.hypotheses) / max(1, len(history)),
        'meta_learning_effectiveness': controller.meta_state.norm().item()
    }
    
    return analysis