"""
RL training utilities for the illiquid bond market simulator.

This module provides:
- Rollout collectors for gathering trajectories
- Replay buffers for off-policy algorithms
- Training loop utilities
- Integration with popular RL libraries (SB3, CleanRL)
- Evaluation utilities

Example usage with Stable-Baselines3:
    >>> from illiquid_market_sim import TradingEnv, EnvConfig
    >>> from illiquid_market_sim.rl import make_sb3_env, evaluate_policy
    >>> 
    >>> env = make_sb3_env(EnvConfig())
    >>> # Train with SB3...
    >>> metrics = evaluate_policy(model, env, n_episodes=10)
"""

from illiquid_market_sim.rl.rollout import (
    Trajectory,
    RolloutCollector,
    collect_rollout,
    collect_episodes,
)

from illiquid_market_sim.rl.buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    RolloutBuffer,
)

from illiquid_market_sim.rl.wrappers import (
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    TimeLimit,
    make_sb3_env,
    make_vec_env,
)

from illiquid_market_sim.rl.evaluation import (
    evaluate_policy,
    EvaluationResult,
    benchmark_agent,
)

from illiquid_market_sim.rl.callbacks import (
    EvalCallback,
    CheckpointCallback,
    LoggingCallback,
)

from illiquid_market_sim.rl.offline import (
    DataCollector,
    OfflineDataset,
    DatasetMetadata,
)

__all__ = [
    # Rollout
    "Trajectory",
    "RolloutCollector",
    "collect_rollout",
    "collect_episodes",
    
    # Buffers
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "RolloutBuffer",
    
    # Wrappers
    "NormalizeObservation",
    "NormalizeReward",
    "RecordEpisodeStatistics",
    "TimeLimit",
    "make_sb3_env",
    "make_vec_env",
    
    # Evaluation
    "evaluate_policy",
    "EvaluationResult",
    "benchmark_agent",
    
    # Callbacks
    "EvalCallback",
    "CheckpointCallback",
    "LoggingCallback",
    
    # Offline RL
    "DataCollector",
    "OfflineDataset",
    "DatasetMetadata",
]

