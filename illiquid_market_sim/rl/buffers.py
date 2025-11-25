"""
Replay buffers for off-policy RL algorithms.

This module provides various buffer implementations:
- ReplayBuffer: Standard uniform sampling replay buffer
- PrioritizedReplayBuffer: Prioritized experience replay
- RolloutBuffer: Buffer for on-policy algorithms (PPO, A2C)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from illiquid_market_sim.rl.rollout import Trajectory


class Transition(NamedTuple):
    """A single transition tuple."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    info: Optional[Dict[str, Any]] = None


class BatchTransition(NamedTuple):
    """A batch of transitions."""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """
    Standard replay buffer with uniform sampling.
    
    Stores transitions and samples uniformly at random for training.
    
    Example:
        >>> buffer = ReplayBuffer(capacity=10000, obs_dim=40, action_dim=1)
        >>> buffer.add(obs, action, reward, next_obs, done)
        >>> batch = buffer.sample(batch_size=32)
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        action_dtype: np.dtype = np.float32,
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Observation dimension
            action_dim: Action dimension
            action_dtype: Data type for actions
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Pre-allocate arrays
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.pos = 0
        self.size = 0
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_observation
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add all transitions from a trajectory.
        
        Args:
            trajectory: Trajectory to add
        """
        for t in range(trajectory.length):
            self.add(
                observation=trajectory.observations[t],
                action=trajectory.actions[t],
                reward=trajectory.rewards[t],
                next_observation=trajectory.observations[t + 1],
                done=trajectory.dones[t],
            )
    
    def sample(self, batch_size: int) -> BatchTransition:
        """
        Sample a batch of transitions uniformly.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            BatchTransition containing sampled data
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return BatchTransition(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
        )
    
    def __len__(self) -> int:
        """Number of transitions in buffer."""
        return self.size
    
    def is_full(self) -> bool:
        """Whether buffer is at capacity."""
        return self.size == self.capacity
    
    def clear(self) -> None:
        """Clear all transitions."""
        self.pos = 0
        self.size = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions with probability proportional to their TD error,
    allowing more important experiences to be replayed more frequently.
    
    Reference: Schaul et al. "Prioritized Experience Replay" (2015)
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        action_dtype: np.dtype = np.float32,
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum capacity
            obs_dim: Observation dimension
            action_dim: Action dimension
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (annealed to 1)
            beta_increment: How much to increase beta per sample
            epsilon: Small constant to ensure non-zero priorities
            action_dtype: Data type for actions
        """
        super().__init__(capacity, obs_dim, action_dim, action_dtype)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Priority storage (sum tree would be more efficient for large buffers)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Add transition with priority."""
        super().add(observation, action, reward, next_observation, done)
        
        # New transitions get max priority
        self.priorities[self.pos - 1] = priority if priority is not None else self.max_priority
    
    def sample(self, batch_size: int) -> Tuple[BatchTransition, np.ndarray, np.ndarray]:
        """
        Sample with prioritization.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (batch, indices, importance weights)
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = BatchTransition(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
        )
        
        return batch, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically |TD error|)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority + self.epsilon)


class RolloutBuffer:
    """
    Buffer for on-policy algorithms (PPO, A2C).
    
    Stores complete rollouts and computes returns/advantages for policy
    gradient updates. Unlike replay buffers, this is cleared after each
    policy update.
    
    Example:
        >>> buffer = RolloutBuffer(buffer_size=2048, obs_dim=40, action_dim=1)
        >>> # Collect rollout
        >>> for step in range(2048):
        ...     buffer.add(obs, action, reward, value, log_prob, done)
        >>> # Compute advantages
        >>> buffer.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)
        >>> # Get batches for training
        >>> for batch in buffer.get_batches(batch_size=64):
        ...     # Update policy
        ...     pass
        >>> buffer.reset()
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        action_dtype: np.dtype = np.float32,
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Size of buffer (number of steps)
            obs_dim: Observation dimension
            action_dim: Action dimension
            action_dtype: Data type for actions
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Pre-allocate arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # Computed after rollout
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Add a step to the buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
        """
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute returns and GAE advantages.
        
        Args:
            last_value: Value estimate for the state after the last step
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        last_gae = 0.0
        
        for t in reversed(range(self.pos)):
            if t == self.pos - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + self.values[t]
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        Generate batches for training.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
        
        Yields:
            Dict containing batch data
        """
        indices = np.arange(self.pos)
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, self.pos, batch_size):
            end = min(start + batch_size, self.pos)
            batch_indices = indices[start:end]
            
            yield {
                "observations": self.observations[batch_indices],
                "actions": self.actions[batch_indices],
                "values": self.values[batch_indices],
                "log_probs": self.log_probs[batch_indices],
                "returns": self.returns[batch_indices],
                "advantages": self.advantages[batch_indices],
            }
    
    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all data as a dict."""
        return {
            "observations": self.observations[:self.pos],
            "actions": self.actions[:self.pos],
            "rewards": self.rewards[:self.pos],
            "values": self.values[:self.pos],
            "log_probs": self.log_probs[:self.pos],
            "dones": self.dones[:self.pos],
            "returns": self.returns[:self.pos],
            "advantages": self.advantages[:self.pos],
        }
    
    def reset(self) -> None:
        """Reset buffer for next rollout."""
        self.pos = 0
        self.full = False
    
    def __len__(self) -> int:
        """Current number of steps in buffer."""
        return self.pos

