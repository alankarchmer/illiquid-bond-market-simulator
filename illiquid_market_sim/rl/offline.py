"""
Offline RL data collection and loading utilities.

This module provides tools for:
- Collecting trajectories from policies and saving to disk
- Loading saved datasets for offline RL training
- Dataset versioning and schema validation
- Integration with common offline RL formats

Supported formats:
- NPZ (NumPy compressed)
- JSON (for small datasets)
- Parquet (if pyarrow is available)

Dataset Schema (v1):
    observations: np.ndarray [N, obs_dim]
    actions: np.ndarray [N, action_dim]
    rewards: np.ndarray [N]
    next_observations: np.ndarray [N, obs_dim]
    dones: np.ndarray [N] (bool)
    episode_starts: np.ndarray [N] (bool, marks first step of episodes)
    infos: List[Dict] (optional, stored as JSON)

Example:
    >>> from illiquid_market_sim import TradingEnv, EnvConfig
    >>> from illiquid_market_sim.rl.offline import DataCollector, OfflineDataset
    >>> 
    >>> # Collect data
    >>> env = TradingEnv(EnvConfig())
    >>> collector = DataCollector(env)
    >>> collector.collect_episodes(n_episodes=100, policy=my_policy)
    >>> collector.save("./data/expert_data.npz")
    >>> 
    >>> # Load for training
    >>> dataset = OfflineDataset.load("./data/expert_data.npz")
    >>> for batch in dataset.iterate_batches(batch_size=32):
    ...     # Train offline RL algorithm
    ...     pass
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from illiquid_market_sim.rl.rollout import Trajectory


# Dataset schema version
SCHEMA_VERSION = "1.0"


@dataclass
class DatasetMetadata:
    """Metadata for an offline dataset."""
    schema_version: str = SCHEMA_VERSION
    n_transitions: int = 0
    n_episodes: int = 0
    obs_dim: int = 0
    action_dim: int = 0
    env_config: Dict[str, Any] = field(default_factory=dict)
    collection_policy: str = "unknown"
    creation_date: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_transitions": self.n_transitions,
            "n_episodes": self.n_episodes,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "env_config": self.env_config,
            "collection_policy": self.collection_policy,
            "creation_date": self.creation_date,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DataCollector:
    """
    Collects experience data from an environment.
    
    Stores transitions in memory and provides methods to save to disk.
    
    Example:
        >>> collector = DataCollector(env)
        >>> collector.collect_episodes(100, policy=my_policy)
        >>> print(f"Collected {collector.n_transitions} transitions")
        >>> collector.save("data.npz")
    """
    
    def __init__(
        self,
        env: Any,
        max_transitions: Optional[int] = None,
    ):
        """
        Initialize collector.
        
        Args:
            env: Environment to collect from
            max_transitions: Maximum transitions to store (None = unlimited)
        """
        self.env = env
        self.max_transitions = max_transitions
        
        # Storage
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_observations: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.episode_starts: List[bool] = []
        self.infos: List[Dict[str, Any]] = []
        
        self.n_episodes = 0
        self._is_first_step = True
    
    @property
    def n_transitions(self) -> int:
        """Number of transitions collected."""
        return len(self.rewards)
    
    def collect_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single transition.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
            info: Additional info
        """
        if self.max_transitions and self.n_transitions >= self.max_transitions:
            return
        
        self.observations.append(obs.copy())
        self.actions.append(action.copy() if isinstance(action, np.ndarray) else np.array([action]))
        self.rewards.append(reward)
        self.next_observations.append(next_obs.copy())
        self.dones.append(done)
        self.episode_starts.append(self._is_first_step)
        self.infos.append(info or {})
        
        self._is_first_step = done
        if done:
            self.n_episodes += 1
    
    def collect_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add all transitions from a trajectory.
        
        Args:
            trajectory: Trajectory to add
        """
        for t in range(trajectory.length):
            self.collect_step(
                obs=trajectory.observations[t],
                action=trajectory.actions[t],
                reward=trajectory.rewards[t],
                next_obs=trajectory.observations[t + 1],
                done=trajectory.dones[t],
                info=trajectory.infos[t] if t < len(trajectory.infos) else None,
            )
    
    def collect_episode(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        seed: Optional[int] = None,
    ) -> float:
        """
        Collect a single episode.
        
        Args:
            policy: Policy function (obs -> action)
            seed: Random seed for episode
        
        Returns:
            Total episode reward
        """
        obs, info = self.env.reset(seed=seed)
        episode_reward = 0.0
        done = False
        
        while not done:
            action = policy(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.collect_step(obs, action, reward, next_obs, done, info)
            
            obs = next_obs
            episode_reward += reward
        
        return episode_reward
    
    def collect_episodes(
        self,
        n_episodes: int,
        policy: Callable[[np.ndarray], np.ndarray],
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> List[float]:
        """
        Collect multiple episodes.
        
        Args:
            n_episodes: Number of episodes to collect
            policy: Policy function
            seed: Base random seed
            verbose: Whether to print progress
        
        Returns:
            List of episode rewards
        """
        rewards = []
        
        for i in range(n_episodes):
            ep_seed = seed + i if seed is not None else None
            ep_reward = self.collect_episode(policy, seed=ep_seed)
            rewards.append(ep_reward)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Collected {i + 1}/{n_episodes} episodes, "
                      f"mean reward: {np.mean(rewards):.2f}")
        
        return rewards
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert collected data to numpy arrays."""
        return {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "next_observations": np.array(self.next_observations),
            "dones": np.array(self.dones),
            "episode_starts": np.array(self.episode_starts),
        }
    
    def save(
        self,
        filepath: str,
        format: str = "npz",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save collected data to file.
        
        Args:
            filepath: Path to save to
            format: File format ("npz", "json", "parquet")
            metadata: Additional metadata to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        arrays = self.to_arrays()
        
        # Create metadata
        meta = DatasetMetadata(
            n_transitions=self.n_transitions,
            n_episodes=self.n_episodes,
            obs_dim=arrays["observations"].shape[1] if len(arrays["observations"]) > 0 else 0,
            action_dim=arrays["actions"].shape[1] if len(arrays["actions"]) > 0 else 0,
        )
        if metadata:
            for k, v in metadata.items():
                if hasattr(meta, k):
                    setattr(meta, k, v)
        
        if format == "npz":
            self._save_npz(filepath, arrays, meta)
        elif format == "json":
            self._save_json(filepath, arrays, meta)
        elif format == "parquet":
            self._save_parquet(filepath, arrays, meta)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _save_npz(
        self,
        filepath: Path,
        arrays: Dict[str, np.ndarray],
        metadata: DatasetMetadata
    ) -> None:
        """Save as NPZ file."""
        # Save arrays
        np.savez_compressed(
            filepath,
            **arrays,
            metadata=json.dumps(metadata.to_dict()),
        )
    
    def _save_json(
        self,
        filepath: Path,
        arrays: Dict[str, np.ndarray],
        metadata: DatasetMetadata
    ) -> None:
        """Save as JSON file (for small datasets)."""
        data = {
            "metadata": metadata.to_dict(),
            "observations": arrays["observations"].tolist(),
            "actions": arrays["actions"].tolist(),
            "rewards": arrays["rewards"].tolist(),
            "next_observations": arrays["next_observations"].tolist(),
            "dones": arrays["dones"].tolist(),
            "episode_starts": arrays["episode_starts"].tolist(),
            "infos": self.infos,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
    
    def _save_parquet(
        self,
        filepath: Path,
        arrays: Dict[str, np.ndarray],
        metadata: DatasetMetadata
    ) -> None:
        """Save as Parquet file (requires pyarrow)."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for parquet format. "
                            "Install with: pip install pyarrow")
        
        # Create table
        table = pa.table({
            "observation": [obs.tolist() for obs in arrays["observations"]],
            "action": [act.tolist() for act in arrays["actions"]],
            "reward": arrays["rewards"],
            "next_observation": [obs.tolist() for obs in arrays["next_observations"]],
            "done": arrays["dones"],
            "episode_start": arrays["episode_starts"],
        })
        
        # Save with metadata
        meta_bytes = json.dumps(metadata.to_dict()).encode()
        table = table.replace_schema_metadata({b"dataset_metadata": meta_bytes})
        pq.write_table(table, filepath)
    
    def clear(self) -> None:
        """Clear all collected data."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.episode_starts = []
        self.infos = []
        self.n_episodes = 0
        self._is_first_step = True


class OfflineDataset:
    """
    Dataset for offline RL training.
    
    Provides efficient iteration over stored transitions.
    
    Example:
        >>> dataset = OfflineDataset.load("data.npz")
        >>> print(f"Dataset has {len(dataset)} transitions")
        >>> 
        >>> for batch in dataset.iterate_batches(batch_size=32):
        ...     obs, actions, rewards, next_obs, dones = batch
        ...     # Train your algorithm
    """
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        episode_starts: Optional[np.ndarray] = None,
        metadata: Optional[DatasetMetadata] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            observations: Observations [N, obs_dim]
            actions: Actions [N, action_dim]
            rewards: Rewards [N]
            next_observations: Next observations [N, obs_dim]
            dones: Done flags [N]
            episode_starts: Episode start flags [N]
            metadata: Dataset metadata
        """
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones
        self.episode_starts = episode_starts
        self.metadata = metadata or DatasetMetadata()
    
    def __len__(self) -> int:
        """Number of transitions in dataset."""
        return len(self.rewards)
    
    @property
    def obs_dim(self) -> int:
        """Observation dimension."""
        return self.observations.shape[1] if len(self.observations) > 0 else 0
    
    @property
    def action_dim(self) -> int:
        """Action dimension."""
        return self.actions.shape[1] if len(self.actions) > 0 else 0
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        indices = np.random.randint(0, len(self), size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )
    
    def iterate_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """
        Iterate over dataset in batches.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle before iterating
            drop_last: Whether to drop the last incomplete batch
        
        Yields:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            if end > len(self) and drop_last:
                break
            
            batch_indices = indices[start:end]
            yield (
                self.observations[batch_indices],
                self.actions[batch_indices],
                self.rewards[batch_indices],
                self.next_observations[batch_indices],
                self.dones[batch_indices],
            )
    
    def get_episode(self, episode_idx: int) -> Tuple[np.ndarray, ...]:
        """
        Get a specific episode.
        
        Args:
            episode_idx: Episode index
        
        Returns:
            Tuple of episode data
        """
        if self.episode_starts is None:
            raise ValueError("Episode starts not available in this dataset")
        
        # Find episode boundaries
        episode_start_indices = np.where(self.episode_starts)[0]
        
        if episode_idx >= len(episode_start_indices):
            raise IndexError(f"Episode {episode_idx} not found")
        
        start = episode_start_indices[episode_idx]
        if episode_idx + 1 < len(episode_start_indices):
            end = episode_start_indices[episode_idx + 1]
        else:
            end = len(self)
        
        return (
            self.observations[start:end],
            self.actions[start:end],
            self.rewards[start:end],
            self.next_observations[start:end],
            self.dones[start:end],
        )
    
    @property
    def n_episodes(self) -> int:
        """Number of episodes in dataset."""
        if self.episode_starts is None:
            return 0
        return int(np.sum(self.episode_starts))
    
    @classmethod
    def load(cls, filepath: str) -> "OfflineDataset":
        """
        Load dataset from file.
        
        Args:
            filepath: Path to dataset file
        
        Returns:
            OfflineDataset instance
        """
        filepath = Path(filepath)
        
        if filepath.suffix == ".npz":
            return cls._load_npz(filepath)
        elif filepath.suffix == ".json":
            return cls._load_json(filepath)
        elif filepath.suffix == ".parquet":
            return cls._load_parquet(filepath)
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")
    
    @classmethod
    def _load_npz(cls, filepath: Path) -> "OfflineDataset":
        """Load from NPZ file."""
        data = np.load(filepath, allow_pickle=True)
        
        metadata = None
        if "metadata" in data:
            meta_str = str(data["metadata"])
            metadata = DatasetMetadata.from_dict(json.loads(meta_str))
        
        return cls(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            next_observations=data["next_observations"],
            dones=data["dones"],
            episode_starts=data.get("episode_starts"),
            metadata=metadata,
        )
    
    @classmethod
    def _load_json(cls, filepath: Path) -> "OfflineDataset":
        """Load from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        metadata = DatasetMetadata.from_dict(data.get("metadata", {}))
        
        return cls(
            observations=np.array(data["observations"]),
            actions=np.array(data["actions"]),
            rewards=np.array(data["rewards"]),
            next_observations=np.array(data["next_observations"]),
            dones=np.array(data["dones"]),
            episode_starts=np.array(data.get("episode_starts", [])),
            metadata=metadata,
        )
    
    @classmethod
    def _load_parquet(cls, filepath: Path) -> "OfflineDataset":
        """Load from Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for parquet format")
        
        table = pq.read_table(filepath)
        
        # Extract metadata
        metadata = None
        if b"dataset_metadata" in table.schema.metadata:
            meta_str = table.schema.metadata[b"dataset_metadata"].decode()
            metadata = DatasetMetadata.from_dict(json.loads(meta_str))
        
        return cls(
            observations=np.array([list(x) for x in table["observation"].to_pylist()]),
            actions=np.array([list(x) for x in table["action"].to_pylist()]),
            rewards=np.array(table["reward"].to_pylist()),
            next_observations=np.array([list(x) for x in table["next_observation"].to_pylist()]),
            dones=np.array(table["done"].to_pylist()),
            episode_starts=np.array(table["episode_start"].to_pylist()),
            metadata=metadata,
        )
    
    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
    ) -> Tuple["OfflineDataset", "OfflineDataset"]:
        """
        Split dataset into train and test sets.
        
        Args:
            train_ratio: Fraction for training
            shuffle: Whether to shuffle before splitting
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        
        split_idx = int(len(self) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_dataset = OfflineDataset(
            observations=self.observations[train_indices],
            actions=self.actions[train_indices],
            rewards=self.rewards[train_indices],
            next_observations=self.next_observations[train_indices],
            dones=self.dones[train_indices],
        )
        
        test_dataset = OfflineDataset(
            observations=self.observations[test_indices],
            actions=self.actions[test_indices],
            rewards=self.rewards[test_indices],
            next_observations=self.next_observations[test_indices],
            dones=self.dones[test_indices],
        )
        
        return train_dataset, test_dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "n_transitions": len(self),
            "n_episodes": self.n_episodes,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "mean_reward": float(np.mean(self.rewards)),
            "std_reward": float(np.std(self.rewards)),
            "min_reward": float(np.min(self.rewards)),
            "max_reward": float(np.max(self.rewards)),
            "mean_episode_length": len(self) / max(self.n_episodes, 1),
        }

