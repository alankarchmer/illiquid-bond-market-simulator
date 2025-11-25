"""
Evaluation utilities for RL agents.

This module provides tools for evaluating trained policies:
- Policy evaluation over multiple episodes
- Benchmark comparisons against baseline agents
- Performance metrics and statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from illiquid_market_sim.rl.rollout import Trajectory, RolloutCollector


@dataclass
class EvaluationResult:
    """
    Results from policy evaluation.
    
    Attributes:
        n_episodes: Number of episodes evaluated
        mean_reward: Mean episode reward
        std_reward: Standard deviation of episode rewards
        min_reward: Minimum episode reward
        max_reward: Maximum episode reward
        mean_length: Mean episode length
        mean_trades: Mean number of trades per episode
        mean_pnl: Mean final PnL
        mean_fill_ratio: Mean RFQ fill ratio
        episode_rewards: List of individual episode rewards
        episode_infos: List of final info dicts from each episode
    """
    n_episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_length: float
    mean_trades: float = 0.0
    mean_pnl: float = 0.0
    mean_fill_ratio: float = 0.0
    episode_rewards: List[float] = field(default_factory=list)
    episode_infos: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_episodes": self.n_episodes,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "mean_length": self.mean_length,
            "mean_trades": self.mean_trades,
            "mean_pnl": self.mean_pnl,
            "mean_fill_ratio": self.mean_fill_ratio,
        }
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Evaluation Results",
            "=" * 40,
            f"Episodes: {self.n_episodes}",
            f"Mean Reward: {self.mean_reward:.2f} (+/- {self.std_reward:.2f})",
            f"Min/Max Reward: {self.min_reward:.2f} / {self.max_reward:.2f}",
            f"Mean Length: {self.mean_length:.1f}",
            f"Mean Trades: {self.mean_trades:.1f}",
            f"Mean PnL: {self.mean_pnl:.2f}",
            f"Mean Fill Ratio: {self.mean_fill_ratio:.1%}",
        ]
        return "\n".join(lines)


def evaluate_policy(
    env: Any,
    policy: Callable[[np.ndarray], np.ndarray],
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    seed: Optional[int] = None,
) -> EvaluationResult:
    """
    Evaluate a policy over multiple episodes.
    
    Args:
        env: Environment to evaluate in
        policy: Policy function (obs -> action)
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        render: Whether to render episodes
        seed: Random seed for reproducibility
    
    Returns:
        EvaluationResult with statistics
    
    Example:
        >>> from illiquid_market_sim import TradingEnv, EnvConfig
        >>> env = TradingEnv(EnvConfig())
        >>> 
        >>> def my_policy(obs):
        ...     # Your trained policy
        ...     return np.array([0.5])
        >>> 
        >>> result = evaluate_policy(env, my_policy, n_episodes=10)
        >>> print(result.summary())
    """
    episode_rewards = []
    episode_lengths = []
    episode_trades = []
    episode_pnls = []
    episode_fill_ratios = []
    episode_infos = []
    
    for ep in range(n_episodes):
        # Reset with seed if provided
        ep_seed = seed + ep if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_infos.append(info)
        
        # Extract additional metrics from info
        if "total_trades" in info:
            episode_trades.append(info["total_trades"])
        if "total_pnl" in info:
            episode_pnls.append(info["total_pnl"])
        if "fill_ratio" in info:
            episode_fill_ratios.append(info["fill_ratio"])
    
    return EvaluationResult(
        n_episodes=n_episodes,
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        min_reward=float(np.min(episode_rewards)),
        max_reward=float(np.max(episode_rewards)),
        mean_length=float(np.mean(episode_lengths)),
        mean_trades=float(np.mean(episode_trades)) if episode_trades else 0.0,
        mean_pnl=float(np.mean(episode_pnls)) if episode_pnls else 0.0,
        mean_fill_ratio=float(np.mean(episode_fill_ratios)) if episode_fill_ratios else 0.0,
        episode_rewards=episode_rewards,
        episode_infos=episode_infos,
    )


@dataclass
class BenchmarkResult:
    """Results from benchmarking multiple agents."""
    agent_results: Dict[str, EvaluationResult]
    
    def get_ranking(self, metric: str = "mean_reward") -> List[Tuple[str, float]]:
        """Get agents ranked by a metric."""
        scores = []
        for name, result in self.agent_results.items():
            score = getattr(result, metric, 0.0)
            scores.append((name, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def summary(self) -> str:
        """Get comparison summary."""
        lines = ["Benchmark Results", "=" * 60]
        
        # Header
        lines.append(f"{'Agent':<20} {'Mean Reward':>12} {'Std':>8} {'PnL':>10} {'Fill%':>8}")
        lines.append("-" * 60)
        
        # Sort by mean reward
        ranking = self.get_ranking("mean_reward")
        
        for name, _ in ranking:
            result = self.agent_results[name]
            lines.append(
                f"{name:<20} "
                f"{result.mean_reward:>12.2f} "
                f"{result.std_reward:>8.2f} "
                f"{result.mean_pnl:>10.2f} "
                f"{result.mean_fill_ratio:>7.1%}"
            )
        
        return "\n".join(lines)


def benchmark_agent(
    env: Any,
    agents: Dict[str, Callable[[np.ndarray], np.ndarray]],
    n_episodes: int = 10,
    seed: Optional[int] = None,
) -> BenchmarkResult:
    """
    Benchmark multiple agents against each other.
    
    Args:
        env: Environment to evaluate in
        agents: Dict mapping agent name to policy function
        n_episodes: Number of episodes per agent
        seed: Random seed for reproducibility
    
    Returns:
        BenchmarkResult with comparison
    
    Example:
        >>> from illiquid_market_sim import TradingEnv, EnvConfig
        >>> env = TradingEnv(EnvConfig())
        >>> 
        >>> agents = {
        ...     "random": lambda obs: env.action_space_sample(),
        ...     "conservative": lambda obs: np.array([0.8]),  # Wide spreads
        ...     "aggressive": lambda obs: np.array([0.2]),    # Tight spreads
        ... }
        >>> 
        >>> result = benchmark_agent(env, agents, n_episodes=10)
        >>> print(result.summary())
    """
    results = {}
    
    for name, policy in agents.items():
        result = evaluate_policy(
            env=env,
            policy=policy,
            n_episodes=n_episodes,
            seed=seed,
        )
        results[name] = result
    
    return BenchmarkResult(agent_results=results)


def compute_metrics(
    trajectories: List[Trajectory],
) -> Dict[str, float]:
    """
    Compute detailed metrics from trajectories.
    
    Args:
        trajectories: List of collected trajectories
    
    Returns:
        Dict of metrics
    """
    if not trajectories:
        return {}
    
    rewards = [t.total_reward for t in trajectories]
    lengths = [t.length for t in trajectories]
    
    # Compute returns if not already computed
    all_returns = []
    for t in trajectories:
        if t.returns is None:
            t.compute_returns()
        all_returns.extend(t.returns.tolist())
    
    metrics = {
        # Episode-level
        "mean_episode_reward": np.mean(rewards),
        "std_episode_reward": np.std(rewards),
        "min_episode_reward": np.min(rewards),
        "max_episode_reward": np.max(rewards),
        "mean_episode_length": np.mean(lengths),
        
        # Step-level
        "mean_step_reward": np.mean([r for t in trajectories for r in t.rewards]),
        "mean_return": np.mean(all_returns),
        
        # Totals
        "total_episodes": len(trajectories),
        "total_steps": sum(lengths),
    }
    
    return metrics


def compute_trading_metrics(
    env: Any,
    trajectories: List[Trajectory],
) -> Dict[str, float]:
    """
    Compute trading-specific metrics from trajectories.
    
    Args:
        env: Environment (for accessing additional info)
        trajectories: List of collected trajectories
    
    Returns:
        Dict of trading metrics
    """
    if not trajectories:
        return {}
    
    # Extract from final infos
    pnls = []
    fill_ratios = []
    trade_counts = []
    inventory_risks = []
    
    for t in trajectories:
        if t.infos:
            final_info = t.infos[-1]
            if "total_pnl" in final_info:
                pnls.append(final_info["total_pnl"])
            if "fill_ratio" in final_info:
                fill_ratios.append(final_info["fill_ratio"])
            if "total_trades" in final_info:
                trade_counts.append(final_info["total_trades"])
            if "inventory_risk" in final_info:
                inventory_risks.append(final_info["inventory_risk"])
    
    metrics = {}
    
    if pnls:
        metrics["mean_pnl"] = np.mean(pnls)
        metrics["std_pnl"] = np.std(pnls)
        metrics["sharpe"] = np.mean(pnls) / (np.std(pnls) + 1e-8)
        
        # Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        metrics["max_drawdown"] = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    if fill_ratios:
        metrics["mean_fill_ratio"] = np.mean(fill_ratios)
    
    if trade_counts:
        metrics["mean_trades"] = np.mean(trade_counts)
    
    if inventory_risks:
        metrics["mean_inventory_risk"] = np.mean(inventory_risks)
    
    return metrics

