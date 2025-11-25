"""
Baseline agents for benchmarking RL algorithms.

This module provides several heuristic trading strategies that can be used
as baselines for comparison with learned policies:

1. RandomAgent: Uniform random spread selection
2. FixedSpreadAgent: Always quotes a fixed spread
3. InventoryAwareAgent: Widens spread based on inventory
4. AdaptiveAgent: Adjusts spread based on market conditions
5. ConservativeAgent: Wide spreads, low risk
6. AggressiveAgent: Tight spreads, high volume

Each agent implements a simple policy function that maps observations to actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


class BaselineAgent(ABC):
    """
    Base class for baseline trading agents.
    
    Subclasses implement the `get_action` method which maps
    observations to actions.
    """
    
    def __init__(self, name: str = "baseline"):
        self.name = name
        self._step_count = 0
    
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action given observation.
        
        Args:
            observation: Current observation from environment
        
        Returns:
            Action array
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state (called at episode start)."""
        self._step_count = 0
    
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Allow agent to be used as a callable policy."""
        action = self.get_action(observation)
        self._step_count += 1
        return action
    
    def get_policy(self) -> Callable[[np.ndarray], np.ndarray]:
        """Get a policy function for this agent."""
        return self.__call__


class RandomAgent(BaselineAgent):
    """
    Agent that selects random spreads.
    
    Useful as a lower bound for performance comparison.
    """
    
    def __init__(self, action_dim: int = 1, seed: Optional[int] = None):
        super().__init__(name="random")
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return self.rng.uniform(0, 1, size=self.action_dim).astype(np.float32)


class FixedSpreadAgent(BaselineAgent):
    """
    Agent that always quotes a fixed spread.
    
    Args:
        spread_level: Spread level in [0, 1], where 0 = min spread, 1 = max spread
    """
    
    def __init__(self, spread_level: float = 0.5):
        super().__init__(name=f"fixed_{spread_level:.2f}")
        self.spread_level = np.clip(spread_level, 0, 1)
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return np.array([self.spread_level], dtype=np.float32)


class InventoryAwareAgent(BaselineAgent):
    """
    Agent that adjusts spread based on current inventory.
    
    Widens spread when inventory is high to reduce risk.
    This mimics basic market maker behavior.
    
    Args:
        base_spread: Base spread level when inventory is zero
        inventory_sensitivity: How much to widen per unit of inventory
        max_spread: Maximum spread level
    """
    
    def __init__(
        self,
        base_spread: float = 0.3,
        inventory_sensitivity: float = 0.1,
        max_spread: float = 0.9,
    ):
        super().__init__(name="inventory_aware")
        self.base_spread = base_spread
        self.inventory_sensitivity = inventory_sensitivity
        self.max_spread = max_spread
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        # Extract inventory from observation
        # Based on ObservationSpec, position is at index 17 (in bond features)
        # and inventory_risk is at index 20 (in portfolio features)
        
        # Use inventory risk as a proxy (index 20, normalized by 50)
        inventory_risk = observation[20] * 50.0 if len(observation) > 20 else 0.0
        
        # Widen spread based on inventory
        spread = self.base_spread + self.inventory_sensitivity * abs(inventory_risk)
        spread = min(spread, self.max_spread)
        
        return np.array([spread], dtype=np.float32)


class AdaptiveAgent(BaselineAgent):
    """
    Agent that adapts to market conditions.
    
    Considers:
    - Market volatility/stress (wider spreads in stressed markets)
    - Bond liquidity (wider spreads for illiquid bonds)
    - Client type (wider spreads for informed traders)
    - Inventory (wider spreads when inventory is high)
    
    This is a more sophisticated heuristic that combines multiple signals.
    """
    
    def __init__(
        self,
        base_spread: float = 0.3,
        volatility_weight: float = 0.2,
        liquidity_weight: float = 0.2,
        inventory_weight: float = 0.15,
        client_weight: float = 0.15,
    ):
        super().__init__(name="adaptive")
        self.base_spread = base_spread
        self.volatility_weight = volatility_weight
        self.liquidity_weight = liquidity_weight
        self.inventory_weight = inventory_weight
        self.client_weight = client_weight
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        spread = self.base_spread
        
        if len(observation) < 30:
            return np.array([spread], dtype=np.float32)
        
        # Market stress indicator (index 27)
        stress = observation[27] if len(observation) > 27 else 0.0
        spread += self.volatility_weight * stress
        
        # Bond liquidity (index 11, inverted - low liquidity = wider spread)
        liquidity = observation[11] if len(observation) > 11 else 0.5
        spread += self.liquidity_weight * (1.0 - liquidity)
        
        # Inventory risk (index 20)
        inventory_risk = abs(observation[20]) if len(observation) > 20 else 0.0
        spread += self.inventory_weight * inventory_risk
        
        # Client type - hedge funds get wider spreads (index 3)
        is_hedge_fund = observation[3] if len(observation) > 3 else 0.0
        spread += self.client_weight * is_hedge_fund
        
        # Clip to valid range
        spread = np.clip(spread, 0.1, 0.95)
        
        return np.array([spread], dtype=np.float32)


class ConservativeAgent(BaselineAgent):
    """
    Conservative agent that quotes wide spreads.
    
    Prioritizes low risk over volume. Good for volatile markets
    or when learning about client behavior.
    """
    
    def __init__(self, min_spread: float = 0.6, max_spread: float = 0.9):
        super().__init__(name="conservative")
        self.min_spread = min_spread
        self.max_spread = max_spread
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        # Base conservative spread
        spread = self.min_spread
        
        # Widen further if market is stressed or inventory is high
        if len(observation) > 27:
            stress = observation[27]
            spread += 0.1 * stress
        
        if len(observation) > 20:
            inventory = abs(observation[20])
            spread += 0.1 * inventory
        
        spread = np.clip(spread, self.min_spread, self.max_spread)
        
        return np.array([spread], dtype=np.float32)


class AggressiveAgent(BaselineAgent):
    """
    Aggressive agent that quotes tight spreads.
    
    Prioritizes volume over per-trade profit. Good for liquid markets
    with predictable client behavior.
    """
    
    def __init__(self, min_spread: float = 0.1, max_spread: float = 0.4):
        super().__init__(name="aggressive")
        self.min_spread = min_spread
        self.max_spread = max_spread
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        # Base aggressive spread
        spread = self.min_spread
        
        # Only widen if absolutely necessary (high inventory or stress)
        if len(observation) > 20:
            inventory = abs(observation[20])
            if inventory > 0.5:  # Only react to significant inventory
                spread += 0.15 * inventory
        
        if len(observation) > 27:
            stress = observation[27]
            if stress > 0.5:  # Only react to significant stress
                spread += 0.1 * stress
        
        spread = np.clip(spread, self.min_spread, self.max_spread)
        
        return np.array([spread], dtype=np.float32)


class TrendFollowingAgent(BaselineAgent):
    """
    Agent that adjusts spread based on recent PnL trend.
    
    Widens spread after losses, tightens after gains.
    This is a simple momentum-based strategy.
    """
    
    def __init__(
        self,
        base_spread: float = 0.4,
        trend_sensitivity: float = 0.2,
        lookback: int = 5,
    ):
        super().__init__(name="trend_following")
        self.base_spread = base_spread
        self.trend_sensitivity = trend_sensitivity
        self.lookback = lookback
        self._pnl_history: list = []
    
    def reset(self) -> None:
        super().reset()
        self._pnl_history = []
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        spread = self.base_spread
        
        # Get current PnL from observation (index 18)
        current_pnl = observation[18] * 100.0 if len(observation) > 18 else 0.0
        self._pnl_history.append(current_pnl)
        
        # Calculate trend
        if len(self._pnl_history) >= 2:
            recent = self._pnl_history[-self.lookback:]
            if len(recent) >= 2:
                trend = recent[-1] - recent[0]
                
                # Negative trend (losing) -> widen spread
                # Positive trend (winning) -> tighten spread
                spread -= self.trend_sensitivity * np.tanh(trend / 10.0)
        
        spread = np.clip(spread, 0.1, 0.9)
        
        return np.array([spread], dtype=np.float32)


class ClientTieringAgent(BaselineAgent):
    """
    Agent that quotes different spreads based on client type.
    
    This mimics real-world dealer behavior where different client
    types receive different pricing:
    - Real money: Tight spreads (good flow)
    - Noise traders: Tight spreads (uninformed)
    - Hedge funds: Wide spreads (informed)
    - Fishers: Very wide spreads (just fishing)
    """
    
    def __init__(
        self,
        real_money_spread: float = 0.25,
        noise_spread: float = 0.30,
        hedge_fund_spread: float = 0.55,
        fisher_spread: float = 0.70,
        unknown_spread: float = 0.45,
    ):
        super().__init__(name="client_tiering")
        self.spreads = {
            "real_money": real_money_spread,
            "noise": noise_spread,
            "hedge_fund": hedge_fund_spread,
            "fisher": fisher_spread,
            "unknown": unknown_spread,
        }
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        # Detect client type from observation (indices 2-5)
        if len(observation) < 6:
            spread = self.spreads["unknown"]
        else:
            if observation[2] > 0.5:
                spread = self.spreads["real_money"]
            elif observation[3] > 0.5:
                spread = self.spreads["hedge_fund"]
            elif observation[4] > 0.5:
                spread = self.spreads["fisher"]
            elif observation[5] > 0.5:
                spread = self.spreads["noise"]
            else:
                spread = self.spreads["unknown"]
        
        # Adjust for inventory
        if len(observation) > 20:
            inventory = abs(observation[20])
            spread += 0.1 * inventory
        
        spread = np.clip(spread, 0.1, 0.9)
        
        return np.array([spread], dtype=np.float32)


# -----------------------------------------------------------------------------
# Benchmark Tasks
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkTask:
    """
    A benchmark task specification.
    
    Defines a specific scenario for evaluating agents.
    """
    name: str
    description: str
    env_config: Dict[str, Any]
    n_episodes: int = 10
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "env_config": self.env_config,
            "n_episodes": self.n_episodes,
            "seed": self.seed,
        }


# Standard benchmark tasks
BENCHMARK_TASKS = {
    "normal_market": BenchmarkTask(
        name="normal_market",
        description="Normal market conditions with moderate volatility",
        env_config={
            "sim_config": {
                "num_bonds": 50,
                "num_steps": 100,
                "market_volatility": 0.02,
                "jump_probability": 0.05,
            }
        },
        n_episodes=20,
    ),
    
    "stressed_market": BenchmarkTask(
        name="stressed_market",
        description="Stressed market with high volatility and frequent jumps",
        env_config={
            "sim_config": {
                "num_bonds": 50,
                "num_steps": 100,
                "market_volatility": 0.05,
                "jump_probability": 0.15,
            }
        },
        n_episodes=20,
    ),
    
    "illiquid_market": BenchmarkTask(
        name="illiquid_market",
        description="Illiquid market with high impact and wide spreads",
        env_config={
            "sim_config": {
                "num_bonds": 30,
                "num_steps": 100,
                "base_impact_coeff": 0.005,
                "rfq_prob_per_client": 0.05,
            }
        },
        n_episodes=20,
    ),
    
    "high_volume": BenchmarkTask(
        name="high_volume",
        description="High volume market with many RFQs",
        env_config={
            "sim_config": {
                "num_bonds": 50,
                "num_steps": 100,
                "rfq_prob_per_client": 0.20,
                "num_real_money_clients": 5,
                "num_hedge_fund_clients": 3,
            }
        },
        n_episodes=20,
    ),
    
    "toxic_flow": BenchmarkTask(
        name="toxic_flow",
        description="Market with many informed traders",
        env_config={
            "sim_config": {
                "num_bonds": 50,
                "num_steps": 100,
                "num_hedge_fund_clients": 5,
                "num_fisher_clients": 4,
                "num_real_money_clients": 1,
            }
        },
        n_episodes=20,
    ),
}


def get_all_baseline_agents() -> Dict[str, BaselineAgent]:
    """
    Get all baseline agents for benchmarking.
    
    Returns:
        Dict mapping agent name to agent instance
    """
    return {
        "random": RandomAgent(),
        "fixed_tight": FixedSpreadAgent(0.2),
        "fixed_medium": FixedSpreadAgent(0.5),
        "fixed_wide": FixedSpreadAgent(0.8),
        "inventory_aware": InventoryAwareAgent(),
        "adaptive": AdaptiveAgent(),
        "conservative": ConservativeAgent(),
        "aggressive": AggressiveAgent(),
        "trend_following": TrendFollowingAgent(),
        "client_tiering": ClientTieringAgent(),
    }


def run_benchmark(
    env: Any,
    agents: Optional[Dict[str, BaselineAgent]] = None,
    tasks: Optional[Dict[str, BenchmarkTask]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run benchmark evaluation across agents and tasks.
    
    Args:
        env: Base environment (will be reconfigured per task)
        agents: Dict of agents to evaluate (uses all baselines if None)
        tasks: Dict of tasks to run (uses all standard tasks if None)
        verbose: Whether to print progress
    
    Returns:
        Nested dict: {task_name: {agent_name: {metric: value}}}
    """
    from illiquid_market_sim.rl.evaluation import evaluate_policy
    
    agents = agents or get_all_baseline_agents()
    tasks = tasks or BENCHMARK_TASKS
    
    results = {}
    
    for task_name, task in tasks.items():
        if verbose:
            print(f"\nRunning task: {task_name}")
            print(f"  {task.description}")
        
        results[task_name] = {}
        
        for agent_name, agent in agents.items():
            if verbose:
                print(f"  Evaluating: {agent_name}...", end=" ")
            
            agent.reset()
            
            eval_result = evaluate_policy(
                env=env,
                policy=agent.get_policy(),
                n_episodes=task.n_episodes,
                seed=task.seed,
            )
            
            results[task_name][agent_name] = eval_result.to_dict()
            
            if verbose:
                print(f"mean_reward={eval_result.mean_reward:.2f}")
    
    return results


def print_benchmark_results(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    for task_name, task_results in results.items():
        print(f"\n{task_name.upper()}")
        print("-" * 60)
        print(f"{'Agent':<20} {'Mean Reward':>12} {'Std':>8} {'PnL':>10}")
        print("-" * 60)
        
        # Sort by mean reward
        sorted_agents = sorted(
            task_results.items(),
            key=lambda x: x[1].get("mean_reward", 0),
            reverse=True
        )
        
        for agent_name, metrics in sorted_agents:
            print(
                f"{agent_name:<20} "
                f"{metrics.get('mean_reward', 0):>12.2f} "
                f"{metrics.get('std_reward', 0):>8.2f} "
                f"{metrics.get('mean_pnl', 0):>10.2f}"
            )

