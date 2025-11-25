"""
Rollout collection utilities for RL training.

This module provides tools for collecting trajectories from the environment,
which can be used for both on-policy and off-policy RL algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Trajectory:
    """
    A single trajectory (episode) of experience.
    
    Attributes:
        observations: Array of observations [T+1, obs_dim]
        actions: Array of actions [T, action_dim]
        rewards: Array of rewards [T]
        dones: Array of done flags [T]
        infos: List of info dicts [T]
        returns: Computed returns (if computed)
        advantages: Computed advantages (if computed)
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, Any]] = field(default_factory=list)
    returns: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None
    
    @property
    def length(self) -> int:
        """Number of transitions in trajectory."""
        return len(self.rewards)
    
    @property
    def total_reward(self) -> float:
        """Sum of rewards in trajectory."""
        return float(np.sum(self.rewards))
    
    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """
        Compute discounted returns.
        
        Args:
            gamma: Discount factor
        
        Returns:
            Array of returns for each timestep
        """
        returns = np.zeros_like(self.rewards)
        running_return = 0.0
        
        for t in reversed(range(len(self.rewards))):
            running_return = self.rewards[t] + gamma * running_return * (1 - self.dones[t])
            returns[t] = running_return
        
        self.returns = returns
        return returns
    
    def compute_gae(
        self,
        values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            values: Value estimates for each state [T+1]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            Array of advantages for each timestep
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(self.rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
        
        self.advantages = advantages
        self.returns = advantages + values[:-1]
        return advantages
    
    def to_transitions(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        """
        Convert to list of (s, a, r, s', done) transitions.
        
        Returns:
            List of transition tuples
        """
        transitions = []
        for t in range(len(self.rewards)):
            transitions.append((
                self.observations[t],
                self.actions[t],
                self.rewards[t],
                self.observations[t + 1],
                bool(self.dones[t])
            ))
        return transitions


class RolloutCollector:
    """
    Collects rollouts from an environment.
    
    Supports both deterministic and stochastic policies, and can collect
    either a fixed number of steps or complete episodes.
    
    Example:
        >>> from illiquid_market_sim import TradingEnv, EnvConfig
        >>> env = TradingEnv(EnvConfig())
        >>> collector = RolloutCollector(env)
        >>> 
        >>> # Collect with random policy
        >>> trajectory = collector.collect_episode()
        >>> 
        >>> # Collect with custom policy
        >>> def my_policy(obs):
        ...     return env.action_space_sample()
        >>> trajectory = collector.collect_episode(policy=my_policy)
    """
    
    def __init__(
        self,
        env: Any,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize collector.
        
        Args:
            env: Environment to collect from
            policy: Policy function (obs -> action). Uses random if None.
        """
        self.env = env
        self.policy = policy
        
        # For tracking across episodes
        self._total_steps = 0
        self._total_episodes = 0
    
    def collect_episode(
        self,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        render: bool = False,
    ) -> Trajectory:
        """
        Collect a single episode.
        
        Args:
            policy: Policy to use (overrides default)
            render: Whether to render environment
        
        Returns:
            Trajectory containing the episode
        """
        policy = policy or self.policy or self._random_policy
        
        observations = []
        actions = []
        rewards = []
        dones = []
        infos = []
        
        obs, info = self.env.reset()
        observations.append(obs)
        
        done = False
        while not done:
            if render:
                self.env.render()
            
            action = policy(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            self._total_steps += 1
        
        self._total_episodes += 1
        
        return Trajectory(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            infos=infos,
        )
    
    def collect_steps(
        self,
        n_steps: int,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[List[Trajectory], int]:
        """
        Collect at least n_steps across potentially multiple episodes.
        
        Args:
            n_steps: Minimum number of steps to collect
            policy: Policy to use
        
        Returns:
            Tuple of (list of trajectories, total steps collected)
        """
        policy = policy or self.policy or self._random_policy
        
        trajectories = []
        total_steps = 0
        
        while total_steps < n_steps:
            trajectory = self.collect_episode(policy=policy)
            trajectories.append(trajectory)
            total_steps += trajectory.length
        
        return trajectories, total_steps
    
    def collect_episodes(
        self,
        n_episodes: int,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> List[Trajectory]:
        """
        Collect n complete episodes.
        
        Args:
            n_episodes: Number of episodes to collect
            policy: Policy to use
        
        Returns:
            List of trajectories
        """
        policy = policy or self.policy or self._random_policy
        
        trajectories = []
        for _ in range(n_episodes):
            trajectory = self.collect_episode(policy=policy)
            trajectories.append(trajectory)
        
        return trajectories
    
    def _random_policy(self, obs: np.ndarray) -> np.ndarray:
        """Random policy for exploration."""
        return self.env.action_space_sample()
    
    @property
    def total_steps(self) -> int:
        """Total steps collected so far."""
        return self._total_steps
    
    @property
    def total_episodes(self) -> int:
        """Total episodes collected so far."""
        return self._total_episodes


def collect_rollout(
    env: Any,
    policy: Callable[[np.ndarray], np.ndarray],
    n_steps: int,
) -> Tuple[List[Trajectory], Dict[str, float]]:
    """
    Convenience function to collect rollout data.
    
    Args:
        env: Environment
        policy: Policy function
        n_steps: Number of steps to collect
    
    Returns:
        Tuple of (trajectories, statistics dict)
    """
    collector = RolloutCollector(env, policy)
    trajectories, total_steps = collector.collect_steps(n_steps)
    
    stats = {
        "n_episodes": len(trajectories),
        "n_steps": total_steps,
        "mean_reward": np.mean([t.total_reward for t in trajectories]),
        "std_reward": np.std([t.total_reward for t in trajectories]),
        "mean_length": np.mean([t.length for t in trajectories]),
    }
    
    return trajectories, stats


def collect_episodes(
    env: Any,
    policy: Callable[[np.ndarray], np.ndarray],
    n_episodes: int,
) -> Tuple[List[Trajectory], Dict[str, float]]:
    """
    Convenience function to collect complete episodes.
    
    Args:
        env: Environment
        policy: Policy function
        n_episodes: Number of episodes to collect
    
    Returns:
        Tuple of (trajectories, statistics dict)
    """
    collector = RolloutCollector(env, policy)
    trajectories = collector.collect_episodes(n_episodes)
    
    stats = {
        "n_episodes": len(trajectories),
        "n_steps": sum(t.length for t in trajectories),
        "mean_reward": np.mean([t.total_reward for t in trajectories]),
        "std_reward": np.std([t.total_reward for t in trajectories]),
        "mean_length": np.mean([t.length for t in trajectories]),
    }
    
    return trajectories, stats

