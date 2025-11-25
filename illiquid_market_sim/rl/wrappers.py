"""
Environment wrappers for RL training.

This module provides wrappers that modify environment behavior:
- Observation normalization
- Reward normalization/scaling
- Episode statistics recording
- Time limits
- Vectorized environment creation
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from illiquid_market_sim.env import TradingEnv, EnvConfig
from illiquid_market_sim.spaces import RunningNormalizer


class BaseWrapper:
    """Base class for environment wrappers."""
    
    def __init__(self, env: Any):
        self.env = env
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class NormalizeObservation(BaseWrapper):
    """
    Wrapper that normalizes observations using running statistics.
    
    Tracks mean and variance of observations online and normalizes
    new observations accordingly.
    
    Example:
        >>> env = TradingEnv(EnvConfig())
        >>> env = NormalizeObservation(env)
        >>> obs, _ = env.reset()  # Normalized observation
    """
    
    def __init__(
        self,
        env: Any,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        training: bool = True,
    ):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            epsilon: Small constant for numerical stability
            clip: Clip normalized observations to [-clip, clip]
            training: Whether to update statistics (set False for evaluation)
        """
        super().__init__(env)
        
        # Get observation dimension
        obs_dim = env.observation_space_def.total_dim
        
        self.normalizer = RunningNormalizer(
            shape=(obs_dim,),
            epsilon=epsilon,
            clip=clip,
        )
        self.training = training
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        
        if self.training:
            self.normalizer.update(obs)
        
        return self.normalizer.normalize(obs), info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.training:
            self.normalizer.update(obs)
        
        return self.normalizer.normalize(obs), reward, terminated, truncated, info
    
    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self.training = training
    
    def get_state(self) -> Dict[str, Any]:
        """Get normalizer state for saving."""
        return self.normalizer.state_dict()
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Load normalizer state."""
        self.normalizer.load_state_dict(state)


class NormalizeReward(BaseWrapper):
    """
    Wrapper that normalizes rewards using running statistics.
    
    Normalizes rewards based on the running variance of returns,
    which helps with training stability.
    """
    
    def __init__(
        self,
        env: Any,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        training: bool = True,
    ):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            gamma: Discount factor for return computation
            epsilon: Small constant for numerical stability
            clip: Clip normalized rewards to [-clip, clip]
            training: Whether to update statistics
        """
        super().__init__(env)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.training = training
        
        self.return_rms = RunningNormalizer(shape=(), epsilon=epsilon, clip=None)
        self.returns = 0.0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.returns = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.training:
            self.returns = self.returns * self.gamma + reward
            self.return_rms.update(np.array([self.returns]))
        
        # Normalize reward
        normalized_reward = reward / (np.sqrt(self.return_rms.var + self.epsilon))
        normalized_reward = np.clip(normalized_reward, -self.clip, self.clip)
        
        if terminated or truncated:
            self.returns = 0.0
        
        # Store original reward in info
        info["original_reward"] = reward
        
        return obs, float(normalized_reward), terminated, truncated, info
    
    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self.training = training


class RecordEpisodeStatistics(BaseWrapper):
    """
    Wrapper that records episode statistics.
    
    Tracks episode returns, lengths, and other metrics.
    """
    
    def __init__(self, env: Any, buffer_size: int = 100):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            buffer_size: Number of episodes to keep in history
        """
        super().__init__(env)
        
        self.buffer_size = buffer_size
        
        # Current episode stats
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_trades = 0
        
        # History buffers
        self.return_history: List[float] = []
        self.length_history: List[int] = []
        self.trade_history: List[int] = []
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Record completed episode if any
        if self.episode_length > 0:
            self._record_episode()
        
        # Reset current episode stats
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_trades = 0
        
        return self.env.reset(**kwargs)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_return += reward
        self.episode_length += 1
        
        if "total_trades" in info:
            self.episode_trades = info["total_trades"]
        
        if terminated or truncated:
            self._record_episode()
            info["episode"] = {
                "r": self.episode_return,
                "l": self.episode_length,
                "t": self.episode_trades,
            }
        
        return obs, reward, terminated, truncated, info
    
    def _record_episode(self) -> None:
        """Record episode to history."""
        self.return_history.append(self.episode_return)
        self.length_history.append(self.episode_length)
        self.trade_history.append(self.episode_trades)
        
        # Trim to buffer size
        if len(self.return_history) > self.buffer_size:
            self.return_history = self.return_history[-self.buffer_size:]
            self.length_history = self.length_history[-self.buffer_size:]
            self.trade_history = self.trade_history[-self.buffer_size:]
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics over recent episodes."""
        if not self.return_history:
            return {}
        
        return {
            "mean_return": np.mean(self.return_history),
            "std_return": np.std(self.return_history),
            "min_return": np.min(self.return_history),
            "max_return": np.max(self.return_history),
            "mean_length": np.mean(self.length_history),
            "mean_trades": np.mean(self.trade_history),
            "n_episodes": len(self.return_history),
        }


class TimeLimit(BaseWrapper):
    """
    Wrapper that enforces a time limit on episodes.
    
    Truncates episodes after a maximum number of steps.
    """
    
    def __init__(self, env: Any, max_episode_steps: int):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            max_episode_steps: Maximum steps per episode
        """
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._elapsed_steps += 1
        
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True
            info["TimeLimit.truncated"] = True
        
        return obs, reward, terminated, truncated, info


class FrameStack(BaseWrapper):
    """
    Wrapper that stacks consecutive observations.
    
    Useful for providing temporal context to the agent.
    """
    
    def __init__(self, env: Any, n_frames: int = 4):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            n_frames: Number of frames to stack
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames: List[np.ndarray] = []
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        
        # Initialize frame buffer with copies of first observation
        self.frames = [obs.copy() for _ in range(self.n_frames)]
        
        return self._get_stacked_obs(), info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add new frame and remove oldest
        self.frames.pop(0)
        self.frames.append(obs.copy())
        
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        """Stack frames into single observation."""
        return np.concatenate(self.frames, axis=-1)


# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------

def make_sb3_env(
    config: Optional[EnvConfig] = None,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    record_stats: bool = True,
) -> Any:
    """
    Create an environment compatible with Stable-Baselines3.
    
    Args:
        config: Environment configuration
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        record_stats: Whether to record episode statistics
    
    Returns:
        Wrapped environment
    """
    env = TradingEnv(config)
    
    if record_stats:
        env = RecordEpisodeStatistics(env)
    
    if normalize_obs:
        env = NormalizeObservation(env)
    
    if normalize_reward:
        env = NormalizeReward(env)
    
    return env


def make_vec_env(
    config: Optional[EnvConfig] = None,
    n_envs: int = 4,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
) -> List[Any]:
    """
    Create multiple environments for parallel training.
    
    Note: This creates independent environments. For true vectorization,
    use SB3's SubprocVecEnv or DummyVecEnv.
    
    Args:
        config: Environment configuration
        n_envs: Number of environments
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
    
    Returns:
        List of wrapped environments
    """
    envs = []
    
    for i in range(n_envs):
        # Create config with different seed for each env
        env_config = config or EnvConfig()
        if env_config.seed is not None:
            env_config.seed = env_config.seed + i
        
        env = make_sb3_env(
            config=env_config,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
        )
        envs.append(env)
    
    return envs

