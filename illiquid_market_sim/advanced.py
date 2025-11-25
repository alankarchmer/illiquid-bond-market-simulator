"""
Advanced RL features for the illiquid bond market simulator.

This module provides:
- Curriculum learning utilities
- Adversarial/stress test generators
- Multi-objective reward functions
- Domain randomization

These features help with:
- Improving sample efficiency through curriculum learning
- Testing robustness through adversarial scenarios
- Balancing multiple objectives (PnL, risk, client satisfaction)
- Generalization through domain randomization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from illiquid_market_sim.config import SimulationConfig, MarketRegime, get_regime_config
from illiquid_market_sim.env import EnvConfig, TradingEnv


# -----------------------------------------------------------------------------
# Curriculum Learning
# -----------------------------------------------------------------------------

class CurriculumStage(str, Enum):
    """Stages in the curriculum."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class CurriculumConfig:
    """
    Configuration for curriculum learning.
    
    Attributes:
        stages: List of curriculum stages with their configs
        promotion_threshold: Performance threshold to advance
        demotion_threshold: Performance threshold to go back
        window_size: Number of episodes to average for promotion decision
        auto_advance: Whether to automatically advance based on performance
    """
    stages: List[Tuple[CurriculumStage, EnvConfig]] = field(default_factory=list)
    promotion_threshold: float = 0.8
    demotion_threshold: float = 0.3
    window_size: int = 20
    auto_advance: bool = True
    
    def __post_init__(self):
        if not self.stages:
            self.stages = self._default_stages()
    
    @staticmethod
    def _default_stages() -> List[Tuple[CurriculumStage, EnvConfig]]:
        """Create default curriculum stages."""
        return [
            (CurriculumStage.BEGINNER, EnvConfig(
                max_episode_steps=50,
                sim_config=SimulationConfig(
                    num_bonds=20,
                    market_volatility=0.01,
                    jump_probability=0.02,
                    num_hedge_fund_clients=1,
                    num_fisher_clients=1,
                ),
            )),
            (CurriculumStage.INTERMEDIATE, EnvConfig(
                max_episode_steps=75,
                sim_config=SimulationConfig(
                    num_bonds=35,
                    market_volatility=0.02,
                    jump_probability=0.05,
                ),
            )),
            (CurriculumStage.ADVANCED, EnvConfig(
                max_episode_steps=100,
                sim_config=SimulationConfig(
                    num_bonds=50,
                    market_volatility=0.03,
                    jump_probability=0.08,
                    num_hedge_fund_clients=3,
                ),
            )),
            (CurriculumStage.EXPERT, EnvConfig(
                max_episode_steps=150,
                sim_config=SimulationConfig(
                    num_bonds=75,
                    market_volatility=0.05,
                    jump_probability=0.12,
                    num_hedge_fund_clients=4,
                    num_fisher_clients=4,
                ),
            )),
        ]


class CurriculumScheduler:
    """
    Manages curriculum progression during training.
    
    Tracks performance and advances/demotes through curriculum stages
    based on recent episode performance.
    
    Example:
        >>> scheduler = CurriculumScheduler()
        >>> env = scheduler.get_current_env()
        >>> 
        >>> for episode in range(1000):
        ...     reward = run_episode(env)
        ...     if scheduler.update(reward):
        ...         env = scheduler.get_current_env()
        ...         print(f"Advanced to {scheduler.current_stage}")
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """
        Initialize curriculum scheduler.
        
        Args:
            config: Curriculum configuration
        """
        self.config = config or CurriculumConfig()
        self.current_stage_idx = 0
        self.episode_rewards: List[float] = []
        self.stage_history: List[CurriculumStage] = []
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Current curriculum stage."""
        return self.config.stages[self.current_stage_idx][0]
    
    @property
    def current_config(self) -> EnvConfig:
        """Current environment config."""
        return self.config.stages[self.current_stage_idx][1]
    
    def get_current_env(self) -> TradingEnv:
        """Create environment for current stage."""
        return TradingEnv(self.current_config)
    
    def update(self, episode_reward: float) -> bool:
        """
        Update with episode reward and check for stage change.
        
        Args:
            episode_reward: Reward from completed episode
        
        Returns:
            True if stage changed, False otherwise
        """
        self.episode_rewards.append(episode_reward)
        
        if not self.config.auto_advance:
            return False
        
        if len(self.episode_rewards) < self.config.window_size:
            return False
        
        # Calculate normalized performance
        recent = self.episode_rewards[-self.config.window_size:]
        performance = self._calculate_performance(recent)
        
        # Check for promotion
        if performance >= self.config.promotion_threshold:
            if self.current_stage_idx < len(self.config.stages) - 1:
                self.current_stage_idx += 1
                self.stage_history.append(self.current_stage)
                self.episode_rewards = []  # Reset for new stage
                return True
        
        # Check for demotion
        if performance <= self.config.demotion_threshold:
            if self.current_stage_idx > 0:
                self.current_stage_idx -= 1
                self.stage_history.append(self.current_stage)
                self.episode_rewards = []
                return True
        
        return False
    
    def _calculate_performance(self, rewards: List[float]) -> float:
        """Calculate normalized performance score."""
        if not rewards:
            return 0.5
        
        # Normalize based on stage-specific expected performance
        mean_reward = np.mean(rewards)
        
        # Simple normalization - can be made more sophisticated
        # Assumes rewards roughly in [-10, 10] range
        normalized = (mean_reward + 10) / 20
        return np.clip(normalized, 0, 1)
    
    def advance(self) -> bool:
        """Manually advance to next stage."""
        if self.current_stage_idx < len(self.config.stages) - 1:
            self.current_stage_idx += 1
            self.stage_history.append(self.current_stage)
            return True
        return False
    
    def reset(self) -> None:
        """Reset to first stage."""
        self.current_stage_idx = 0
        self.episode_rewards = []
        self.stage_history = []


# -----------------------------------------------------------------------------
# Stress Testing / Adversarial Scenarios
# -----------------------------------------------------------------------------

class StressScenario(str, Enum):
    """Types of stress scenarios."""
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    TOXIC_FLOW = "toxic_flow"
    SECTOR_ROTATION = "sector_rotation"
    SPREAD_WIDENING = "spread_widening"
    RANDOM_WALK = "random_walk"


@dataclass
class StressEvent:
    """A single stress event that can be injected."""
    scenario: StressScenario
    start_step: int
    duration: int
    intensity: float  # 0-1, how severe
    parameters: Dict[str, Any] = field(default_factory=dict)


class StressGenerator:
    """
    Generates adversarial stress scenarios for robustness testing.
    
    Can inject various types of market stress:
    - Flash crashes
    - Liquidity crises
    - Toxic client flow
    - Sector rotations
    - Spread widening events
    
    Example:
        >>> generator = StressGenerator()
        >>> events = generator.generate_episode_events(episode_length=100)
        >>> for event in events:
        ...     print(f"Stress event: {event.scenario} at step {event.start_step}")
    """
    
    def __init__(
        self,
        event_probability: float = 0.1,
        max_events_per_episode: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize stress generator.
        
        Args:
            event_probability: Probability of stress event per step
            max_events_per_episode: Maximum events per episode
            seed: Random seed
        """
        self.event_probability = event_probability
        self.max_events_per_episode = max_events_per_episode
        self.rng = np.random.default_rng(seed)
    
    def generate_episode_events(
        self,
        episode_length: int,
        scenarios: Optional[List[StressScenario]] = None,
    ) -> List[StressEvent]:
        """
        Generate stress events for an episode.
        
        Args:
            episode_length: Length of the episode
            scenarios: Specific scenarios to use (random if None)
        
        Returns:
            List of stress events
        """
        scenarios = scenarios or list(StressScenario)
        events = []
        
        step = 0
        while step < episode_length and len(events) < self.max_events_per_episode:
            if self.rng.random() < self.event_probability:
                scenario = self.rng.choice(scenarios)
                duration = self.rng.integers(5, 20)
                intensity = self.rng.uniform(0.3, 1.0)
                
                event = StressEvent(
                    scenario=scenario,
                    start_step=step,
                    duration=duration,
                    intensity=intensity,
                    parameters=self._generate_parameters(scenario, intensity),
                )
                events.append(event)
                
                # Skip ahead past this event
                step += duration
            else:
                step += 1
        
        return events
    
    def _generate_parameters(
        self,
        scenario: StressScenario,
        intensity: float
    ) -> Dict[str, Any]:
        """Generate scenario-specific parameters."""
        if scenario == StressScenario.FLASH_CRASH:
            return {
                "price_drop": intensity * 10,  # Up to 10% drop
                "recovery_speed": self.rng.uniform(0.1, 0.5),
            }
        
        elif scenario == StressScenario.LIQUIDITY_CRISIS:
            return {
                "liquidity_multiplier": 1 - intensity * 0.8,  # Up to 80% reduction
                "spread_multiplier": 1 + intensity * 3,  # Up to 4x spreads
            }
        
        elif scenario == StressScenario.TOXIC_FLOW:
            return {
                "informed_ratio": intensity * 0.8,  # Up to 80% informed flow
                "hedge_fund_multiplier": 1 + intensity * 2,
            }
        
        elif scenario == StressScenario.SECTOR_ROTATION:
            return {
                "sector_from": self.rng.choice(["IG", "HY", "EM"]),
                "sector_to": self.rng.choice(["IG", "HY", "EM"]),
                "rotation_speed": intensity,
            }
        
        elif scenario == StressScenario.SPREAD_WIDENING:
            return {
                "spread_multiplier": 1 + intensity * 4,  # Up to 5x spreads
            }
        
        elif scenario == StressScenario.RANDOM_WALK:
            return {
                "volatility_multiplier": 1 + intensity * 5,  # Up to 6x volatility
            }
        
        return {}
    
    def apply_event(
        self,
        event: StressEvent,
        current_step: int,
        env_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply stress event modifications to environment state.
        
        Args:
            event: The stress event
            current_step: Current simulation step
            env_state: Current environment state
        
        Returns:
            Modified environment state
        """
        if current_step < event.start_step:
            return env_state
        
        if current_step >= event.start_step + event.duration:
            return env_state
        
        # Calculate progress through event
        progress = (current_step - event.start_step) / event.duration
        
        # Apply modifications based on scenario
        modified = env_state.copy()
        
        if event.scenario == StressScenario.FLASH_CRASH:
            # V-shaped recovery
            if progress < 0.5:
                price_impact = -event.parameters["price_drop"] * (progress * 2)
            else:
                recovery = event.parameters["recovery_speed"]
                price_impact = -event.parameters["price_drop"] * (1 - (progress - 0.5) * 2 * recovery)
            modified["price_adjustment"] = price_impact
        
        elif event.scenario == StressScenario.LIQUIDITY_CRISIS:
            modified["liquidity_multiplier"] = event.parameters["liquidity_multiplier"]
            modified["spread_multiplier"] = event.parameters["spread_multiplier"]
        
        # Add more scenario handlers as needed
        
        return modified


# -----------------------------------------------------------------------------
# Multi-Objective Rewards
# -----------------------------------------------------------------------------

class ObjectiveType(str, Enum):
    """Types of objectives for multi-objective optimization."""
    PNL = "pnl"
    RISK = "risk"
    CLIENT_SATISFACTION = "client_satisfaction"
    EXECUTION_QUALITY = "execution_quality"
    INVENTORY = "inventory"


@dataclass
class MultiObjectiveConfig:
    """
    Configuration for multi-objective rewards.
    
    Attributes:
        objectives: List of objectives to optimize
        weights: Weights for each objective (for scalarization)
        normalize: Whether to normalize each objective
        pareto: Whether to return vector rewards (for MORL)
    """
    objectives: List[ObjectiveType] = field(default_factory=lambda: [
        ObjectiveType.PNL,
        ObjectiveType.RISK,
        ObjectiveType.CLIENT_SATISFACTION,
    ])
    weights: Dict[ObjectiveType, float] = field(default_factory=lambda: {
        ObjectiveType.PNL: 0.5,
        ObjectiveType.RISK: 0.3,
        ObjectiveType.CLIENT_SATISFACTION: 0.2,
    })
    normalize: bool = True
    pareto: bool = False


class MultiObjectiveReward:
    """
    Computes multi-objective rewards.
    
    Supports:
    - Linear scalarization (weighted sum)
    - Vector rewards for Pareto-based MORL
    - Dynamic weight adjustment
    
    Example:
        >>> mo_reward = MultiObjectiveReward()
        >>> reward = mo_reward.compute(
        ...     pnl_change=1.5,
        ...     inventory_risk=5.0,
        ...     fill_ratio=0.4,
        ...     edge_captured=0.2,
        ... )
    """
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        """
        Initialize multi-objective reward.
        
        Args:
            config: Configuration for objectives and weights
        """
        self.config = config or MultiObjectiveConfig()
        
        # Running statistics for normalization
        self._stats: Dict[ObjectiveType, Dict[str, float]] = {
            obj: {"mean": 0.0, "std": 1.0, "count": 0}
            for obj in ObjectiveType
        }
    
    def compute(
        self,
        pnl_change: float = 0.0,
        inventory_risk: float = 0.0,
        fill_ratio: float = 0.0,
        edge_captured: float = 0.0,
        **kwargs
    ) -> float | np.ndarray:
        """
        Compute multi-objective reward.
        
        Args:
            pnl_change: Change in PnL
            inventory_risk: Current inventory risk
            fill_ratio: RFQ fill ratio
            edge_captured: Edge captured on trades
        
        Returns:
            Scalar reward (if not pareto) or vector reward
        """
        # Compute individual objectives
        objectives = {}
        
        if ObjectiveType.PNL in self.config.objectives:
            objectives[ObjectiveType.PNL] = pnl_change
        
        if ObjectiveType.RISK in self.config.objectives:
            # Negative because we want to minimize risk
            objectives[ObjectiveType.RISK] = -inventory_risk
        
        if ObjectiveType.CLIENT_SATISFACTION in self.config.objectives:
            # Higher fill ratio = happier clients
            objectives[ObjectiveType.CLIENT_SATISFACTION] = fill_ratio
        
        if ObjectiveType.EXECUTION_QUALITY in self.config.objectives:
            objectives[ObjectiveType.EXECUTION_QUALITY] = edge_captured
        
        if ObjectiveType.INVENTORY in self.config.objectives:
            # Penalty for holding inventory
            objectives[ObjectiveType.INVENTORY] = -abs(inventory_risk)
        
        # Normalize if configured
        if self.config.normalize:
            for obj_type, value in objectives.items():
                objectives[obj_type] = self._normalize(obj_type, value)
        
        # Return vector or scalarized
        if self.config.pareto:
            return np.array([objectives.get(obj, 0.0) for obj in self.config.objectives])
        
        # Linear scalarization
        total = 0.0
        for obj_type, value in objectives.items():
            weight = self.config.weights.get(obj_type, 0.0)
            total += weight * value
        
        return total
    
    def _normalize(self, obj_type: ObjectiveType, value: float) -> float:
        """Normalize objective value using running statistics."""
        stats = self._stats[obj_type]
        
        # Update running statistics
        stats["count"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        stats["std"] = np.sqrt(
            ((stats["count"] - 1) * stats["std"]**2 + delta * (value - stats["mean"]))
            / stats["count"]
        )
        
        # Normalize
        if stats["std"] > 1e-8:
            return (value - stats["mean"]) / stats["std"]
        return value - stats["mean"]
    
    def set_weights(self, weights: Dict[ObjectiveType, float]) -> None:
        """Update objective weights."""
        self.config.weights.update(weights)
    
    def get_objective_values(
        self,
        pnl_change: float = 0.0,
        inventory_risk: float = 0.0,
        fill_ratio: float = 0.0,
        edge_captured: float = 0.0,
    ) -> Dict[ObjectiveType, float]:
        """Get individual objective values without scalarization."""
        return {
            ObjectiveType.PNL: pnl_change,
            ObjectiveType.RISK: -inventory_risk,
            ObjectiveType.CLIENT_SATISFACTION: fill_ratio,
            ObjectiveType.EXECUTION_QUALITY: edge_captured,
            ObjectiveType.INVENTORY: -abs(inventory_risk),
        }


# -----------------------------------------------------------------------------
# Domain Randomization
# -----------------------------------------------------------------------------

@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization."""
    
    # Randomization ranges (min, max multipliers)
    volatility_range: Tuple[float, float] = (0.5, 2.0)
    liquidity_range: Tuple[float, float] = (0.5, 1.5)
    impact_range: Tuple[float, float] = (0.5, 2.0)
    spread_range: Tuple[float, float] = (0.7, 1.5)
    client_ratio_range: Tuple[float, float] = (0.5, 2.0)
    
    # Which parameters to randomize
    randomize_volatility: bool = True
    randomize_liquidity: bool = True
    randomize_impact: bool = True
    randomize_spreads: bool = True
    randomize_clients: bool = True


class DomainRandomizer:
    """
    Applies domain randomization to environment configs.
    
    Helps with generalization by varying environment parameters
    during training.
    
    Example:
        >>> randomizer = DomainRandomizer()
        >>> base_config = EnvConfig()
        >>> randomized = randomizer.randomize(base_config)
        >>> env = TradingEnv(randomized)
    """
    
    def __init__(
        self,
        config: Optional[DomainRandomizationConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize domain randomizer.
        
        Args:
            config: Randomization configuration
            seed: Random seed
        """
        self.config = config or DomainRandomizationConfig()
        self.rng = np.random.default_rng(seed)
    
    def randomize(self, base_config: EnvConfig) -> EnvConfig:
        """
        Apply randomization to a base config.
        
        Args:
            base_config: Base environment configuration
        
        Returns:
            Randomized configuration
        """
        # Create a copy of the sim config
        sim_config = SimulationConfig(**base_config.sim_config.to_dict())
        
        if self.config.randomize_volatility:
            mult = self.rng.uniform(*self.config.volatility_range)
            sim_config.market_volatility *= mult
        
        if self.config.randomize_impact:
            mult = self.rng.uniform(*self.config.impact_range)
            sim_config.base_impact_coeff *= mult
        
        if self.config.randomize_spreads:
            mult = self.rng.uniform(*self.config.spread_range)
            sim_config.base_spread_bps *= mult
        
        if self.config.randomize_clients:
            mult = self.rng.uniform(*self.config.client_ratio_range)
            sim_config.num_hedge_fund_clients = max(1, int(sim_config.num_hedge_fund_clients * mult))
            sim_config.num_fisher_clients = max(1, int(sim_config.num_fisher_clients * mult))
        
        # Create new env config with randomized sim config
        return EnvConfig(
            sim_config=sim_config,
            max_episode_steps=base_config.max_episode_steps,
            reward_type=base_config.reward_type,
            action_type=base_config.action_type,
            normalize_obs=base_config.normalize_obs,
            normalize_reward=base_config.normalize_reward,
        )
    
    def get_randomized_env(self, base_config: EnvConfig) -> TradingEnv:
        """Create a randomized environment."""
        randomized_config = self.randomize(base_config)
        return TradingEnv(randomized_config)

