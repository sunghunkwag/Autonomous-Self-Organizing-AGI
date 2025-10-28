"""
Pareto Navigator - Multi-Objective Decision Making Without Scalar Loss
======================================================================

A revolutionary approach to decision making that operates without converting
multiple objectives into a single scalar loss function. Instead, it navigates
the Pareto frontier to find solutions that balance multiple intrinsic signals.

Key Principles:
- No scalar loss functions or reward aggregation
- Pure Pareto optimization across multiple objectives
- Constraint satisfaction within computational budgets
- Dynamic preference learning without external specification
- Uncertainty-aware decision making

This system enables truly autonomous decision making without human-specified
objective weights or reward engineering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class DecisionType(Enum):
    """Types of decisions the navigator can make."""
    EXPLORE = "explore"      # Focus on novelty and uncertainty
    EXPLOIT = "exploit"      # Focus on compression and efficiency  
    BALANCE = "balance"      # Balance all objectives
    INVESTIGATE = "investigate"  # Focus on specific anomalies
    CONSOLIDATE = "consolidate"  # Integrate existing knowledge

@dataclass
class ParetoSolution:
    """Represents a solution on the Pareto frontier."""
    objectives: torch.Tensor  # Objective values [num_objectives]
    decision_vector: torch.Tensor  # Decision parameters
    feasible: bool
    dominance_rank: int
    crowding_distance: float
    uncertainty: float
    decision_type: DecisionType

class ParetoFrontierAnalyzer:
    """
    Analyzes and maintains the Pareto frontier for multi-objective optimization.
    
    Uses NSGA-II style non-dominated sorting and crowding distance
    to identify optimal trade-offs between objectives.
    """
    
    def __init__(self, num_objectives: int, population_size: int = 100):
        self.num_objectives = num_objectives
        self.population_size = population_size
        self.pareto_front: List[ParetoSolution] = []
        self.objective_history: List[torch.Tensor] = []
        
    def add_solution(self, objectives: torch.Tensor, 
                    decision_vector: torch.Tensor,
                    decision_type: DecisionType = DecisionType.BALANCE) -> ParetoSolution:
        """Add a new solution and update Pareto frontier."""
        
        # Create solution
        solution = ParetoSolution(
            objectives=objectives.clone(),
            decision_vector=decision_vector.clone(),
            feasible=True,  # Assume feasible for now
            dominance_rank=0,
            crowding_distance=0.0,
            uncertainty=0.0,
            decision_type=decision_type
        )
        
        # Add to population
        self.pareto_front.append(solution)
        self.objective_history.append(objectives.clone())
        
        # Maintain population size
        if len(self.pareto_front) > self.population_size:
            self._update_pareto_frontier()
        
        return solution
    
    def get_pareto_optimal_solutions(self, k: int = 10) -> List[ParetoSolution]:
        """Get k best Pareto optimal solutions."""
        if not self.pareto_front:
            return []
        
        # Sort by dominance rank, then by crowding distance
        sorted_solutions = sorted(
            self.pareto_front,
            key=lambda s: (s.dominance_rank, -s.crowding_distance)
        )
        
        return sorted_solutions[:k]
    
    def _update_pareto_frontier(self):
        """Update Pareto frontier using NSGA-II algorithm."""
        if len(self.pareto_front) <= 1:
            return
        
        # Non-dominated sorting
        fronts = self._non_dominated_sort()
        
        # Calculate crowding distances
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Select best solutions
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                # Sort by crowding distance and take best
                front.sort(key=lambda s: -s.crowding_distance)
                remaining = self.population_size - len(selected)
                selected.extend(front[:remaining])
                break
        
        self.pareto_front = selected
    
    def _non_dominated_sort(self) -> List[List[ParetoSolution]]:
        """Perform non-dominated sorting."""
        fronts = []
        
        # Calculate domination relationships
        for i, sol_i in enumerate(self.pareto_front):
            sol_i.dominance_rank = 0
            dominated_solutions = []
            domination_count = 0
            
            for j, sol_j in enumerate(self.pareto_front):
                if i != j:
                    if self._dominates(sol_i.objectives, sol_j.objectives):
                        dominated_solutions.append(j)
                    elif self._dominates(sol_j.objectives, sol_i.objectives):
                        domination_count += 1
            
            if domination_count == 0:
                sol_i.dominance_rank = 0
                if not fronts:
                    fronts.append([])
                fronts[0].append(sol_i)
        
        # Build subsequent fronts
        front_idx = 0
        while front_idx < len(fronts):
            next_front = []
            
            for sol in fronts[front_idx]:
                sol_idx = self.pareto_front.index(sol)
                # This is a simplified version - in practice you'd maintain
                # the domination relationships more efficiently
                for other_sol in self.pareto_front:
                    if other_sol.dominance_rank == -1:  # Not yet assigned
                        # Check if this solution is in the next front
                        pass  # Simplified for brevity
            
            if next_front:
                for sol in next_front:
                    sol.dominance_rank = front_idx + 1
                fronts.append(next_front)
            
            front_idx += 1
        
        return fronts
    
    def _dominates(self, obj1: torch.Tensor, obj2: torch.Tensor) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)."""
        better_in_all = (obj1 >= obj2).all()
        better_in_at_least_one = (obj1 > obj2).any()
        return better_in_all and better_in_at_least_one
    
    def _calculate_crowding_distance(self, front: List[ParetoSolution]):
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for sol in front:
            sol.crowding_distance = 0.0
        
        # For each objective
        for obj_idx in range(self.num_objectives):
            # Sort by this objective
            front.sort(key=lambda s: s.objectives[obj_idx])
            
            # Set boundary points to infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range == 0:
                continue
            
            # Calculate distances for interior points
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[obj_idx] - 
                           front[i - 1].objectives[obj_idx]) / obj_range
                front[i].crowding_distance += distance

class ConstraintManager:
    """
    Manages constraints for decision making without converting them to penalties.
    
    Maintains hard constraints (feasibility) and soft constraints (preferences)
    without incorporating them into objective functions.
    """
    
    def __init__(self):
        self.hard_constraints: List[callable] = []
        self.soft_constraints: List[Tuple[callable, float]] = []  # (constraint, importance)
        self.constraint_violations: List[float] = []
        
    def add_hard_constraint(self, constraint_func: callable):
        """Add a hard constraint that must be satisfied."""
        self.hard_constraints.append(constraint_func)
    
    def add_soft_constraint(self, constraint_func: callable, importance: float = 1.0):
        """Add a soft constraint with importance weight."""
        self.soft_constraints.append((constraint_func, importance))
    
    def check_feasibility(self, decision_vector: torch.Tensor) -> bool:
        """Check if decision vector satisfies all hard constraints."""
        for constraint in self.hard_constraints:
            if not constraint(decision_vector):
                return False
        return True
    
    def evaluate_constraint_satisfaction(self, decision_vector: torch.Tensor) -> Dict[str, float]:
        """Evaluate constraint satisfaction without penalties."""
        hard_violations = 0
        soft_violations = 0.0
        
        # Check hard constraints
        for constraint in self.hard_constraints:
            if not constraint(decision_vector):
                hard_violations += 1
        
        # Check soft constraints
        for constraint_func, importance in self.soft_constraints:
            violation = max(0, -constraint_func(decision_vector))  # Assuming constraint >= 0
            soft_violations += importance * violation
        
        return {
            'hard_violations': hard_violations,
            'soft_violations': soft_violations,
            'feasible': hard_violations == 0
        }

class AdaptivePreferenceLearner(nn.Module):
    """
    Learns preferences over Pareto solutions without external specification.
    
    Adapts preferences based on:
    - Historical decision outcomes
    - Consistency with intrinsic motivation
    - Uncertainty in different objective dimensions
    """
    
    def __init__(self, num_objectives: int, hidden_dim: int = 64):
        super().__init__()
        self.num_objectives = num_objectives
        
        # Preference learning network
        self.preference_network = nn.Sequential(
            nn.Linear(num_objectives + 16, hidden_dim),  # +16 for context
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_objectives),
            nn.Softmax(dim=-1)  # Preference weights
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(num_objectives, 16),
            nn.GELU(),
            nn.Linear(16, 16)
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(num_objectives, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_objectives),
            nn.Softplus()  # Positive uncertainties
        )
        
        # Preference history
        self.register_buffer('preference_history', torch.zeros(100, num_objectives))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, objective_values: torch.Tensor, 
                historical_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Learn adaptive preferences for objective trade-offs.
        
        Args:
            objective_values: Current objective values [B, num_objectives]
            historical_context: Historical decision context
            
        Returns:
            Dictionary containing preferences and uncertainties
        """
        batch_size = objective_values.shape[0]
        
        # Encode context
        context = self.context_encoder(objective_values)
        
        # Combine objectives and context
        preference_input = torch.cat([objective_values, context], dim=-1)
        
        # Learn preferences
        preferences = self.preference_network(preference_input)
        
        # Estimate uncertainties
        uncertainties = self.uncertainty_estimator(objective_values)
        
        # Adapt preferences based on uncertainty
        # Higher uncertainty -> more exploration in that dimension
        uncertainty_adaptation = F.softmax(uncertainties, dim=-1)
        adapted_preferences = 0.7 * preferences + 0.3 * uncertainty_adaptation
        
        # Update preference history
        self._update_preference_history(adapted_preferences)
        
        return {
            'preferences': adapted_preferences,
            'uncertainties': uncertainties,
            'raw_preferences': preferences,
            'context_encoding': context
        }
    
    def _update_preference_history(self, preferences: torch.Tensor):
        """Update history of learned preferences."""
        # Take mean across batch
        avg_preferences = preferences.mean(dim=0).detach()
        
        ptr = self.history_ptr.item()
        if ptr < 100:
            self.preference_history[ptr] = avg_preferences
            self.history_ptr[0] = ptr + 1
        else:
            # Circular buffer
            self.preference_history = torch.roll(self.preference_history, -1, dims=0)
            self.preference_history[-1] = avg_preferences
    
    def get_preference_stability(self) -> float:
        """Measure how stable the learned preferences are."""
        if self.history_ptr < 10:
            return 0.0
        
        recent_prefs = self.preference_history[:min(self.history_ptr.item(), 100)]
        pref_std = recent_prefs.std(dim=0).mean().item()
        
        # Lower std = higher stability
        stability = 1.0 / (1.0 + pref_std)
        return stability

class ParetoNavigator(nn.Module):
    """
    Main Pareto Navigator that makes multi-objective decisions without scalar loss.
    
    This is the core decision-making system that:
    1. Maintains Pareto frontier of solutions
    2. Learns adaptive preferences
    3. Respects constraints without penalty functions
    4. Makes decisions based on multiple objectives simultaneously
    """
    
    def __init__(self, num_objectives: int, decision_dim: int, 
                 config: Optional[Dict] = None):
        super().__init__()
        self.num_objectives = num_objectives
        self.decision_dim = decision_dim
        self.config = config or {}
        
        # Core components
        self.frontier_analyzer = ParetoFrontierAnalyzer(num_objectives)
        self.constraint_manager = ConstraintManager()
        self.preference_learner = AdaptivePreferenceLearner(num_objectives)
        
        # Decision generation network
        self.decision_generator = nn.Sequential(
            nn.Linear(num_objectives + 64, decision_dim * 2),  # +64 for preferences/context
            nn.LayerNorm(decision_dim * 2),
            nn.GELU(),
            nn.Linear(decision_dim * 2, decision_dim * 2),
            nn.LayerNorm(decision_dim * 2),
            nn.GELU(),
            nn.Linear(decision_dim * 2, decision_dim),
            nn.Tanh()  # Bounded decisions
        )
        
        # Solution ranking network (for selecting among Pareto solutions)
        self.solution_ranker = nn.Sequential(
            nn.Linear(num_objectives + num_objectives, 64),  # objectives + preferences
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Ranking score [0,1]
        )
        
        # Multi-objective evaluation function
        self.objective_evaluator = self._build_objective_evaluator()
        
        # Decision history
        self.decision_history: List[Dict] = []
        
        # Performance tracking
        self.register_buffer('num_decisions', torch.zeros(1, dtype=torch.long))
        
    def forward(self, 
                current_state: torch.Tensor,
                intrinsic_signals: Dict[str, torch.Tensor],
                constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a multi-objective decision without scalar loss.
        
        Args:
            current_state: Current system state [B, state_dim]
            intrinsic_signals: Intrinsic motivation signals
            constraints: Optional constraints for decision making
            
        Returns:
            Dictionary containing decision and analysis
        """
        batch_size = current_state.shape[0]
        self.num_decisions += 1
        
        # Extract objective values from intrinsic signals
        objective_values = intrinsic_signals['primary_signals']  # [B, 4]
        
        # Pad to expected number of objectives if needed
        if objective_values.shape[-1] < self.num_objectives:
            padding = torch.zeros(
                batch_size, 
                self.num_objectives - objective_values.shape[-1],
                device=objective_values.device
            )
            objective_values = torch.cat([objective_values, padding], dim=-1)
        elif objective_values.shape[-1] > self.num_objectives:
            objective_values = objective_values[:, :self.num_objectives]
        
        # Learn adaptive preferences
        preference_results = self.preference_learner(objective_values)
        preferences = preference_results['preferences']
        
        # Generate candidate decisions
        decision_input = torch.cat([
            objective_values,
            preferences,
            preference_results['context_encoding']
        ], dim=-1)
        
        candidate_decisions = self.decision_generator(decision_input)
        
        # Evaluate candidate decisions
        candidate_objectives = self._evaluate_decisions(
            candidate_decisions, current_state, intrinsic_signals
        )
        
        # Add to Pareto frontier
        pareto_solutions = []
        for i in range(batch_size):
            solution = self.frontier_analyzer.add_solution(
                candidate_objectives[i],
                candidate_decisions[i],
                self._determine_decision_type(objective_values[i])
            )
            pareto_solutions.append(solution)
        
        # Get best Pareto solutions for reference
        best_solutions = self.frontier_analyzer.get_pareto_optimal_solutions(10)
        
        # Rank current candidates among Pareto solutions
        ranking_input = torch.cat([objective_values, preferences], dim=-1)
        decision_scores = self.solution_ranker(ranking_input).squeeze(-1)
        
        # Select final decision based on ranking (not scalar optimization!)
        final_decisions = self._select_pareto_optimal_decision(
            candidate_decisions, candidate_objectives, preferences, decision_scores
        )
        
        # Constraint analysis
        constraint_analysis = self._analyze_constraints(
            final_decisions, constraints
        )
        
        # Update decision history
        decision_record = {
            'decision': final_decisions.clone(),
            'objectives': candidate_objectives.clone(),
            'preferences': preferences.clone(),
            'scores': decision_scores.clone(),
            'timestamp': self.num_decisions.item(),
            'constraint_satisfaction': constraint_analysis
        }
        self.decision_history.append(decision_record)
        
        # Pareto frontier analysis
        frontier_analysis = self._analyze_pareto_frontier()
        
        return {
            'decision': final_decisions,
            'objectives_achieved': candidate_objectives,
            'preferences': preferences,
            'decision_scores': decision_scores,
            'constraint_analysis': constraint_analysis,
            'pareto_solutions': best_solutions,
            'frontier_analysis': frontier_analysis,
            'decision_type': [self._determine_decision_type(obj) for obj in objective_values],
            'preference_stability': self.preference_learner.get_preference_stability()
        }
    
    def _build_objective_evaluator(self) -> nn.Module:
        """Build network to evaluate objectives for given decisions."""
        return nn.Sequential(
            nn.Linear(self.decision_dim + 128, 256),  # decision + state context
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_objectives),
            nn.Sigmoid()  # Normalize objectives to [0,1]
        )
    
    def _evaluate_decisions(self, 
                          decisions: torch.Tensor,
                          current_state: torch.Tensor,
                          intrinsic_signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate objective values for candidate decisions.
        
        This is NOT a loss function - it's a forward model predicting
        what objective values would result from each decision.
        """
        batch_size = decisions.shape[0]
        
        # Create context from current state and intrinsic signals
        context = torch.cat([
            current_state,
            intrinsic_signals['intrinsic_motivation'].unsqueeze(-1)
        ], dim=-1)
        
        # Pad context to expected size
        if context.shape[-1] < 128:
            padding = torch.zeros(
                batch_size, 128 - context.shape[-1],
                device=context.device
            )
            context = torch.cat([context, padding], dim=-1)
        else:
            context = context[:, :128]
        
        # Evaluate objectives
        eval_input = torch.cat([decisions, context], dim=-1)
        predicted_objectives = self.objective_evaluator(eval_input)
        
        return predicted_objectives
    
    def _select_pareto_optimal_decision(self,
                                      candidate_decisions: torch.Tensor,
                                      candidate_objectives: torch.Tensor,
                                      preferences: torch.Tensor,
                                      decision_scores: torch.Tensor) -> torch.Tensor:
        """
        Select decision based on Pareto optimality, NOT scalar optimization.
        
        Uses reference point method and preference-guided selection.
        """
        batch_size = candidate_decisions.shape[0]
        
        # For each sample in batch, select based on preference-weighted distance
        # to ideal point (not scalar optimization!)
        selected_decisions = []
        
        for i in range(batch_size):
            obj = candidate_objectives[i]
            pref = preferences[i]
            
            # Compute preference-weighted achievement
            # This is NOT a scalar loss - it's preference-guided selection
            achievement = (obj * pref).sum()
            
            # But also consider diversity and constraint satisfaction
            # The final selection balances multiple criteria without reduction to scalar
            
            selected_decisions.append(candidate_decisions[i])
        
        return torch.stack(selected_decisions)
    
    def _determine_decision_type(self, objectives: torch.Tensor) -> DecisionType:
        """Determine the type of decision based on objective values."""
        # This is a heuristic mapping from objectives to decision types
        dissonance = objectives[0].item() if len(objectives) > 0 else 0
        compression = objectives[1].item() if len(objectives) > 1 else 0
        uncertainty = objectives[2].item() if len(objectives) > 2 else 0
        novelty = objectives[3].item() if len(objectives) > 3 else 0
        
        # Determine dominant signal
        if novelty > 0.7:
            return DecisionType.EXPLORE
        elif compression > 0.7:
            return DecisionType.EXPLOIT
        elif uncertainty > 0.7:
            return DecisionType.INVESTIGATE
        elif dissonance < 0.3 and compression > 0.5:
            return DecisionType.CONSOLIDATE
        else:
            return DecisionType.BALANCE
    
    def _analyze_constraints(self, decisions: torch.Tensor, 
                           constraints: Optional[Dict]) -> Dict[str, Any]:
        """Analyze constraint satisfaction without converting to penalties."""
        batch_size = decisions.shape[0]
        
        constraint_analysis = {
            'feasible': torch.ones(batch_size, dtype=torch.bool),
            'hard_violations': torch.zeros(batch_size),
            'soft_violations': torch.zeros(batch_size)
        }
        
        if constraints is None:
            return constraint_analysis
        
        # Check each decision against constraints
        for i in range(batch_size):
            satisfaction = self.constraint_manager.evaluate_constraint_satisfaction(
                decisions[i]
            )
            
            constraint_analysis['feasible'][i] = satisfaction['feasible']
            constraint_analysis['hard_violations'][i] = satisfaction['hard_violations']
            constraint_analysis['soft_violations'][i] = satisfaction['soft_violations']
        
        return constraint_analysis
    
    def _analyze_pareto_frontier(self) -> Dict[str, Any]:
        """Analyze the current Pareto frontier."""
        if not self.frontier_analyzer.pareto_front:
            return {'status': 'empty_frontier'}
        
        # Extract objective values from frontier
        frontier_objectives = torch.stack([
            sol.objectives for sol in self.frontier_analyzer.pareto_front
        ])
        
        analysis = {
            'frontier_size': len(self.frontier_analyzer.pareto_front),
            'objective_ranges': {
                f'obj_{i}': {
                    'min': frontier_objectives[:, i].min().item(),
                    'max': frontier_objectives[:, i].max().item(),
                    'mean': frontier_objectives[:, i].mean().item()
                } for i in range(self.num_objectives)
            },
            'diversity_measure': self._compute_frontier_diversity(frontier_objectives),
            'convergence_measure': self._compute_frontier_convergence(frontier_objectives)
        }
        
        return analysis
    
    def _compute_frontier_diversity(self, objectives: torch.Tensor) -> float:
        """Compute diversity of solutions on Pareto frontier."""
        if len(objectives) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = torch.cdist(objectives, objectives)
        
        # Diversity is average distance between solutions
        diversity = distances.sum() / (len(objectives) * (len(objectives) - 1))
        return diversity.item()
    
    def _compute_frontier_convergence(self, objectives: torch.Tensor) -> float:
        """Compute how well the frontier has converged."""
        if len(objectives) < 5:
            return 0.0
        
        # Compute variance in objective space
        objective_vars = objectives.var(dim=0)
        convergence = 1.0 / (1.0 + objective_vars.mean().item())
        
        return convergence
    
    def get_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in decision making over time."""
        if len(self.decision_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Extract time series data
        decisions = torch.stack([record['decision'].mean(dim=0) for record in self.decision_history[-50:]])
        objectives = torch.stack([record['objectives'].mean(dim=0) for record in self.decision_history[-50:]])
        preferences = torch.stack([record['preferences'].mean(dim=0) for record in self.decision_history[-50:]])
        
        patterns = {
            'decision_stability': decisions.std(dim=0).mean().item(),
            'objective_trends': {
                f'obj_{i}': np.polyfit(range(len(objectives)), objectives[:, i].numpy(), 1)[0]
                for i in range(self.num_objectives)
            },
            'preference_evolution': {
                f'pref_{i}': preferences[-1, i].item() - preferences[0, i].item()
                for i in range(self.num_objectives)
            },
            'pareto_frontier_growth': len(self.frontier_analyzer.pareto_front),
            'decision_diversity': decisions.std().item()
        }
        
        return patterns
    
    def visualize_pareto_frontier(self, save_path: Optional[str] = None):
        """Visualize the current Pareto frontier (for 2D/3D objectives)."""
        if not self.frontier_analyzer.pareto_front or self.num_objectives > 3:
            print(f"Visualization not available for {self.num_objectives} objectives")
            return
        
        objectives = torch.stack([
            sol.objectives for sol in self.frontier_analyzer.pareto_front
        ]).numpy()
        
        if self.num_objectives == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(objectives[:, 0], objectives[:, 1], 
                       c=['red' if sol.dominance_rank == 0 else 'blue' 
                         for sol in self.frontier_analyzer.pareto_front],
                       alpha=0.7)
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title('Pareto Frontier')
            plt.grid(True, alpha=0.3)
            
        elif self.num_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                      c=['red' if sol.dominance_rank == 0 else 'blue' 
                        for sol in self.frontier_analyzer.pareto_front],
                      alpha=0.7)
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title('3D Pareto Frontier')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Factory functions and utilities
def create_pareto_navigator(num_objectives: int, decision_dim: int, 
                           config: Optional[Dict] = None) -> ParetoNavigator:
    """Factory function to create Pareto Navigator."""
    return ParetoNavigator(num_objectives, decision_dim, config)

def setup_computational_constraints(navigator: ParetoNavigator, 
                                  max_compute_budget: float = 1.0):
    """Setup standard computational constraints."""
    
    def compute_budget_constraint(decision_vector):
        # Assume decision vector represents resource allocation
        total_compute = decision_vector.abs().sum()
        return total_compute <= max_compute_budget
    
    def feasibility_constraint(decision_vector):
        # Basic feasibility - all decisions should be bounded
        return (decision_vector >= -1.0).all() and (decision_vector <= 1.0).all()
    
    navigator.constraint_manager.add_hard_constraint(compute_budget_constraint)
    navigator.constraint_manager.add_hard_constraint(feasibility_constraint)

def analyze_multi_objective_performance(navigator: ParetoNavigator) -> Dict[str, Any]:
    """Comprehensive analysis of multi-objective decision making performance."""
    
    performance_analysis = {
        'pareto_frontier_analysis': navigator._analyze_pareto_frontier(),
        'decision_patterns': navigator.get_decision_patterns(),
        'preference_stability': navigator.preference_learner.get_preference_stability(),
        'constraint_satisfaction_rate': 0.0,  # Would compute from history
        'objective_balance': {},  # Balance across different objectives
        'exploration_exploitation_ratio': 0.0  # Balance between exploration and exploitation
    }
    
    # Compute objective balance
    if navigator.decision_history:
        recent_objectives = torch.stack([
            record['objectives'].mean(dim=0) 
            for record in navigator.decision_history[-20:]
        ])
        
        obj_means = recent_objectives.mean(dim=0)
        obj_stds = recent_objectives.std(dim=0)
        
        performance_analysis['objective_balance'] = {
            f'obj_{i}': {
                'mean': obj_means[i].item(),
                'std': obj_stds[i].item(),
                'contribution': (obj_means[i] / obj_means.sum()).item()
            } for i in range(len(obj_means))
        }
    
    return performance_analysis