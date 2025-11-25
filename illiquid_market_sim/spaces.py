"""
Observation, Action, and Reward Space Definitions for RL.

This module provides detailed specifications of the spaces used in the
TradingEnv, along with utilities for normalization, encoding/decoding,
and validation.

Observation Space
-----------------
The observation is a flattened numpy array containing:

1. RFQ Features (8 dims):
   - side: 1.0 for buy, -1.0 for sell
   - size: Normalized trade size
   - client_type_real_money: 1.0 if real money client
   - client_type_hedge_fund: 1.0 if hedge fund client
   - client_type_fisher: 1.0 if fisher client
   - client_type_noise: 1.0 if noise client
   - time_in_episode: Normalized step number
   - rfq_count: Normalized total RFQ count

2. Bond Features (10 dims):
   - sector_IG: 1.0 if IG sector
   - sector_HY: 1.0 if HY sector
   - sector_EM: 1.0 if EM sector
   - liquidity: Bond liquidity score [0, 1]
   - volatility: Normalized volatility
   - maturity: Normalized years to maturity
   - base_spread: Normalized base spread
   - last_price: Last traded price (normalized around 100)
   - naive_mid: Dealer's mid estimate (normalized)
   - current_position: Current position in this bond

3. Portfolio Features (6 dims):
   - total_pnl: Normalized total PnL
   - realized_pnl: Normalized realized PnL
   - inventory_risk: Normalized inventory risk
   - num_positions: Fraction of bonds with positions
   - total_trades: Normalized trade count
   - fill_ratio: RFQ fill ratio

4. Market Features (6 dims):
   - level_IG: IG sector factor level
   - level_HY: HY sector factor level
   - level_EM: EM sector factor level
   - time_in_episode: Normalized step
   - recent_volatility: Recent PnL volatility
   - regime_stress: Market stress indicator

5. History Features (10 dims):
   - recent_pnl_0..4: Last 5 PnL values
   - last_trade_side: Side of last trade
   - last_trade_size: Size of last trade
   - last_trade_price: Price of last trade
   - episode_progress: Fraction of episode complete
   - cumulative_reward: Episode reward so far

Action Space
------------
Multiple action types are supported:

1. CONTINUOUS_SPREAD:
   - Single float in [0, 1]
   - Mapped to [min_spread_bps, max_spread_bps]
   - Agent controls the bid-ask spread to quote

2. DISCRETE_SPREAD:
   - Integer in [0, n_levels-1]
   - Each level corresponds to a spread tier
   - Useful for simpler learning problems

3. CONTINUOUS_PRICE:
   - Single float in [0, 1]
   - Represents price offset from naive mid
   - More direct price control

4. ACCEPT_REJECT:
   - Tuple of (accept: float, spread: float)
   - accept > 0.5 means quote the RFQ
   - spread controls the spread if quoting

Reward Functions
----------------
Multiple reward types are available:

1. PNL:
   - Simple PnL change from previous step
   - reward = new_pnl - old_pnl

2. RISK_ADJUSTED_PNL:
   - PnL minus inventory penalty
   - reward = pnl_change - penalty * inventory_risk

3. EXECUTION_QUALITY:
   - Edge captured vs fair value
   - reward = edge * trade_size (if traded)

4. SHARPE:
   - Rolling Sharpe-like reward
   - reward = pnl_change / rolling_std

5. COMPOSITE:
   - Weighted combination of above
   - Customizable weights

Normalization
-------------
Both observations and rewards can be normalized using running statistics:
- Running mean and standard deviation are tracked
- New values are normalized as: (x - mean) / std
- This helps with neural network training stability

Example Usage
-------------
>>> from illiquid_market_sim.spaces import (
...     get_observation_spec,
...     get_action_spec,
...     normalize_observation,
...     decode_action,
... )
>>> obs_spec = get_observation_spec()
>>> print(f"Observation dim: {obs_spec.total_dim}")
>>> action_spec = get_action_spec(ActionType.CONTINUOUS_SPREAD)
>>> raw_action = np.array([0.5])
>>> decoded = decode_action(raw_action, action_spec)
>>> print(f"Spread: {decoded['spread_bps']:.1f} bps")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class RewardType(str, Enum):
    """Available reward function types."""
    PNL = "pnl"
    RISK_ADJUSTED_PNL = "risk_adjusted_pnl"
    EXECUTION_QUALITY = "execution_quality"
    SHARPE = "sharpe"
    COMPOSITE = "composite"


class ActionType(str, Enum):
    """Available action space types."""
    CONTINUOUS_SPREAD = "continuous_spread"
    DISCRETE_SPREAD = "discrete_spread"
    CONTINUOUS_PRICE = "continuous_price"
    ACCEPT_REJECT = "accept_reject"


# -----------------------------------------------------------------------------
# Observation Space Specification
# -----------------------------------------------------------------------------

@dataclass
class FeatureGroup:
    """Specification for a group of observation features."""
    name: str
    dim: int
    features: List[str]
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    description: str = ""
    
    def __post_init__(self):
        if len(self.features) != self.dim:
            raise ValueError(f"Feature count ({len(self.features)}) != dim ({self.dim})")
        if self.low is None:
            self.low = np.full(self.dim, -np.inf, dtype=np.float32)
        if self.high is None:
            self.high = np.full(self.dim, np.inf, dtype=np.float32)


@dataclass
class ObservationSpec:
    """
    Complete specification of the observation space.
    
    Attributes:
        feature_groups: List of feature groups in order
        total_dim: Total flattened dimension
    """
    feature_groups: List[FeatureGroup] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.feature_groups:
            self.feature_groups = self._default_groups()
    
    @staticmethod
    def _default_groups() -> List[FeatureGroup]:
        """Create default feature groups."""
        return [
            FeatureGroup(
                name="rfq",
                dim=8,
                features=[
                    "side",
                    "size",
                    "client_type_real_money",
                    "client_type_hedge_fund",
                    "client_type_fisher",
                    "client_type_noise",
                    "time_in_episode",
                    "rfq_count",
                ],
                low=np.array([-1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([1, 10, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                description="Current RFQ features",
            ),
            FeatureGroup(
                name="bond",
                dim=10,
                features=[
                    "sector_IG",
                    "sector_HY",
                    "sector_EM",
                    "liquidity",
                    "volatility",
                    "maturity",
                    "base_spread",
                    "last_price",
                    "naive_mid",
                    "current_position",
                ],
                low=np.array([0, 0, 0, 0, 0, 0, 0, -10, -10, -10], dtype=np.float32),
                high=np.array([1, 1, 1, 1, 5, 1, 2, 10, 10, 10], dtype=np.float32),
                description="Bond characteristics",
            ),
            FeatureGroup(
                name="portfolio",
                dim=6,
                features=[
                    "total_pnl",
                    "realized_pnl",
                    "inventory_risk",
                    "num_positions",
                    "total_trades",
                    "fill_ratio",
                ],
                description="Portfolio state",
            ),
            FeatureGroup(
                name="market",
                dim=6,
                features=[
                    "level_IG",
                    "level_HY",
                    "level_EM",
                    "time_in_episode",
                    "recent_volatility",
                    "regime_stress",
                ],
                description="Market state and regime",
            ),
            FeatureGroup(
                name="history",
                dim=10,
                features=[
                    "recent_pnl_0",
                    "recent_pnl_1",
                    "recent_pnl_2",
                    "recent_pnl_3",
                    "recent_pnl_4",
                    "last_trade_side",
                    "last_trade_size",
                    "last_trade_price",
                    "episode_progress",
                    "cumulative_reward",
                ],
                description="Recent history",
            ),
        ]
    
    @property
    def total_dim(self) -> int:
        """Total observation dimension."""
        return sum(g.dim for g in self.feature_groups)
    
    @property
    def all_features(self) -> List[str]:
        """List of all feature names in order."""
        features = []
        for group in self.feature_groups:
            features.extend(group.features)
        return features
    
    def get_feature_index(self, feature_name: str) -> int:
        """Get the index of a feature in the flattened observation."""
        idx = 0
        for group in self.feature_groups:
            if feature_name in group.features:
                return idx + group.features.index(feature_name)
            idx += group.dim
        raise KeyError(f"Feature '{feature_name}' not found")
    
    def get_group_slice(self, group_name: str) -> slice:
        """Get the slice for a feature group in the flattened observation."""
        start = 0
        for group in self.feature_groups:
            if group.name == group_name:
                return slice(start, start + group.dim)
            start += group.dim
        raise KeyError(f"Group '{group_name}' not found")
    
    def get_low(self) -> np.ndarray:
        """Get lower bounds for the observation space."""
        return np.concatenate([g.low for g in self.feature_groups])
    
    def get_high(self) -> np.ndarray:
        """Get upper bounds for the observation space."""
        return np.concatenate([g.high for g in self.feature_groups])
    
    def describe(self) -> str:
        """Get a human-readable description of the observation space."""
        lines = ["Observation Space Specification", "=" * 40, ""]
        
        idx = 0
        for group in self.feature_groups:
            lines.append(f"{group.name.upper()} ({group.dim} dims)")
            lines.append("-" * 40)
            if group.description:
                lines.append(f"  {group.description}")
            for i, feat in enumerate(group.features):
                lines.append(f"  [{idx + i:2d}] {feat}")
            lines.append("")
            idx += group.dim
        
        lines.append(f"Total dimension: {self.total_dim}")
        return "\n".join(lines)


def get_observation_spec() -> ObservationSpec:
    """Get the default observation specification."""
    return ObservationSpec()


# -----------------------------------------------------------------------------
# Action Space Specification
# -----------------------------------------------------------------------------

@dataclass
class ActionSpec:
    """
    Specification of the action space.
    
    Attributes:
        action_type: Type of action space
        dim: Action dimension
        low: Lower bounds
        high: Upper bounds
        n_discrete: Number of discrete levels (for discrete actions)
        min_spread_bps: Minimum spread in basis points
        max_spread_bps: Maximum spread in basis points
    """
    action_type: ActionType
    dim: int
    low: np.ndarray
    high: np.ndarray
    n_discrete: int = 10
    min_spread_bps: float = 10.0
    max_spread_bps: float = 200.0
    
    @property
    def is_discrete(self) -> bool:
        """Whether the action space is discrete."""
        return self.action_type == ActionType.DISCRETE_SPREAD
    
    @property
    def is_continuous(self) -> bool:
        """Whether the action space is continuous."""
        return not self.is_discrete
    
    def describe(self) -> str:
        """Get a human-readable description of the action space."""
        lines = ["Action Space Specification", "=" * 40, ""]
        lines.append(f"Type: {self.action_type.value}")
        lines.append(f"Dimension: {self.dim}")
        
        if self.is_discrete:
            lines.append(f"Discrete levels: {self.n_discrete}")
        else:
            lines.append(f"Bounds: [{self.low}, {self.high}]")
        
        lines.append(f"Spread range: [{self.min_spread_bps}, {self.max_spread_bps}] bps")
        
        return "\n".join(lines)


def get_action_spec(
    action_type: ActionType = ActionType.CONTINUOUS_SPREAD,
    n_discrete: int = 10,
    min_spread_bps: float = 10.0,
    max_spread_bps: float = 200.0,
) -> ActionSpec:
    """
    Get an action specification for the given type.
    
    Args:
        action_type: Type of action space
        n_discrete: Number of discrete levels (for discrete actions)
        min_spread_bps: Minimum spread in basis points
        max_spread_bps: Maximum spread in basis points
    
    Returns:
        ActionSpec for the specified action type
    """
    if action_type == ActionType.CONTINUOUS_SPREAD:
        return ActionSpec(
            action_type=action_type,
            dim=1,
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            n_discrete=n_discrete,
            min_spread_bps=min_spread_bps,
            max_spread_bps=max_spread_bps,
        )
    
    elif action_type == ActionType.DISCRETE_SPREAD:
        return ActionSpec(
            action_type=action_type,
            dim=1,
            low=np.array([0], dtype=np.int32),
            high=np.array([n_discrete - 1], dtype=np.int32),
            n_discrete=n_discrete,
            min_spread_bps=min_spread_bps,
            max_spread_bps=max_spread_bps,
        )
    
    elif action_type == ActionType.CONTINUOUS_PRICE:
        return ActionSpec(
            action_type=action_type,
            dim=1,
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            n_discrete=n_discrete,
            min_spread_bps=min_spread_bps,
            max_spread_bps=max_spread_bps,
        )
    
    elif action_type == ActionType.ACCEPT_REJECT:
        return ActionSpec(
            action_type=action_type,
            dim=2,
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            n_discrete=n_discrete,
            min_spread_bps=min_spread_bps,
            max_spread_bps=max_spread_bps,
        )
    
    raise ValueError(f"Unknown action type: {action_type}")


def decode_action(action: np.ndarray, spec: ActionSpec) -> Dict[str, Any]:
    """
    Decode a raw action into trading parameters.
    
    Args:
        action: Raw action from the policy
        spec: Action specification
    
    Returns:
        Dictionary with 'spread_bps', 'accept', and optionally 'price_offset'
    """
    if spec.action_type == ActionType.CONTINUOUS_SPREAD:
        spread_bps = (
            spec.min_spread_bps +
            float(action[0]) * (spec.max_spread_bps - spec.min_spread_bps)
        )
        return {"spread_bps": spread_bps, "accept": True}
    
    elif spec.action_type == ActionType.DISCRETE_SPREAD:
        level = int(action[0])
        level = max(0, min(level, spec.n_discrete - 1))
        spread_bps = (
            spec.min_spread_bps +
            (level / (spec.n_discrete - 1)) * (spec.max_spread_bps - spec.min_spread_bps)
        )
        return {"spread_bps": spread_bps, "accept": True}
    
    elif spec.action_type == ActionType.CONTINUOUS_PRICE:
        return {"price_offset": float(action[0]), "accept": True}
    
    elif spec.action_type == ActionType.ACCEPT_REJECT:
        accept = float(action[0]) > 0.5
        spread_bps = (
            spec.min_spread_bps +
            float(action[1]) * (spec.max_spread_bps - spec.min_spread_bps)
        )
        return {"spread_bps": spread_bps, "accept": accept}
    
    return {"spread_bps": 50.0, "accept": True}


def encode_action(params: Dict[str, Any], spec: ActionSpec) -> np.ndarray:
    """
    Encode trading parameters into a raw action.
    
    Args:
        params: Dictionary with 'spread_bps' and optionally 'accept'
        spec: Action specification
    
    Returns:
        Raw action array
    """
    spread_bps = params.get("spread_bps", 50.0)
    accept = params.get("accept", True)
    
    # Normalize spread to [0, 1]
    spread_norm = (spread_bps - spec.min_spread_bps) / (spec.max_spread_bps - spec.min_spread_bps)
    spread_norm = max(0.0, min(1.0, spread_norm))
    
    if spec.action_type == ActionType.CONTINUOUS_SPREAD:
        return np.array([spread_norm], dtype=np.float32)
    
    elif spec.action_type == ActionType.DISCRETE_SPREAD:
        level = int(spread_norm * (spec.n_discrete - 1))
        return np.array([level], dtype=np.int32)
    
    elif spec.action_type == ActionType.CONTINUOUS_PRICE:
        price_offset = params.get("price_offset", 0.5)
        return np.array([price_offset], dtype=np.float32)
    
    elif spec.action_type == ActionType.ACCEPT_REJECT:
        accept_val = 1.0 if accept else 0.0
        return np.array([accept_val, spread_norm], dtype=np.float32)
    
    return np.array([spread_norm], dtype=np.float32)


# -----------------------------------------------------------------------------
# Reward Specification
# -----------------------------------------------------------------------------

@dataclass
class RewardSpec:
    """
    Specification of the reward function.
    
    Attributes:
        reward_type: Type of reward function
        scale: Scaling factor for rewards
        inventory_penalty: Penalty coefficient for inventory risk
        composite_weights: Weights for composite reward components
    """
    reward_type: RewardType
    scale: float = 1.0
    inventory_penalty: float = 0.01
    composite_weights: Dict[str, float] = field(default_factory=lambda: {
        "pnl": 0.5,
        "edge": 0.3,
        "inventory": 0.2,
    })
    
    def describe(self) -> str:
        """Get a human-readable description of the reward function."""
        lines = ["Reward Specification", "=" * 40, ""]
        lines.append(f"Type: {self.reward_type.value}")
        lines.append(f"Scale: {self.scale}")
        
        if self.reward_type == RewardType.RISK_ADJUSTED_PNL:
            lines.append(f"Inventory penalty: {self.inventory_penalty}")
        
        if self.reward_type == RewardType.COMPOSITE:
            lines.append("Composite weights:")
            for name, weight in self.composite_weights.items():
                lines.append(f"  {name}: {weight}")
        
        return "\n".join(lines)


def get_reward_spec(
    reward_type: RewardType = RewardType.RISK_ADJUSTED_PNL,
    scale: float = 1.0,
    inventory_penalty: float = 0.01,
) -> RewardSpec:
    """Get a reward specification."""
    return RewardSpec(
        reward_type=reward_type,
        scale=scale,
        inventory_penalty=inventory_penalty,
    )


# -----------------------------------------------------------------------------
# Normalization Utilities
# -----------------------------------------------------------------------------

class RunningNormalizer:
    """
    Running mean and standard deviation normalizer.
    
    Tracks statistics online and normalizes new values accordingly.
    Useful for normalizing observations and rewards during training.
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        epsilon: float = 1e-8,
        clip: Optional[float] = 10.0,
    ):
        """
        Initialize normalizer.
        
        Args:
            shape: Shape of values to normalize
            epsilon: Small constant for numerical stability
            clip: Optional clipping value for normalized outputs
        """
        self.shape = shape
        self.epsilon = epsilon
        self.clip = clip
        
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with new data.
        
        Args:
            x: New data point or batch
        """
        if x.ndim == len(self.shape):
            # Single sample
            batch_mean = x.astype(np.float64)
            batch_var = np.zeros_like(batch_mean)
            batch_count = 1
        else:
            # Batch
            batch_mean = np.mean(x, axis=0, dtype=np.float64)
            batch_var = np.var(x, axis=0, dtype=np.float64)
            batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize input using running statistics.
        
        Args:
            x: Input to normalize
        
        Returns:
            Normalized output
        """
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        
        if self.clip is not None:
            normalized = np.clip(normalized, -self.clip, self.clip)
        
        return normalized.astype(np.float32)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalize input back to original scale.
        
        Args:
            x: Normalized input
        
        Returns:
            Denormalized output
        """
        return (x * np.sqrt(self.var + self.epsilon) + self.mean).astype(np.float32)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for serialization."""
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": self.count,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from serialization."""
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = state["count"]


class ObservationNormalizer:
    """
    Normalizer specifically for observations.
    
    Handles per-feature normalization with feature-specific bounds.
    """
    
    def __init__(self, obs_spec: Optional[ObservationSpec] = None):
        """
        Initialize observation normalizer.
        
        Args:
            obs_spec: Observation specification (uses default if None)
        """
        self.spec = obs_spec or get_observation_spec()
        self.normalizer = RunningNormalizer(shape=(self.spec.total_dim,))
    
    def update(self, obs: np.ndarray) -> None:
        """Update statistics with new observation."""
        self.normalizer.update(obs)
    
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        return self.normalizer.normalize(obs)
    
    def denormalize(self, obs: np.ndarray) -> np.ndarray:
        """Denormalize observation."""
        return self.normalizer.denormalize(obs)
    
    def get_feature(self, obs: np.ndarray, feature_name: str) -> float:
        """Get a specific feature value from an observation."""
        idx = self.spec.get_feature_index(feature_name)
        return float(obs[idx])
    
    def get_group(self, obs: np.ndarray, group_name: str) -> np.ndarray:
        """Get a feature group from an observation."""
        slc = self.spec.get_group_slice(group_name)
        return obs[slc]


class RewardNormalizer:
    """
    Normalizer specifically for rewards.
    
    Tracks reward statistics and normalizes for stable training.
    """
    
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        """
        Initialize reward normalizer.
        
        Args:
            gamma: Discount factor for return normalization
            epsilon: Small constant for numerical stability
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.normalizer = RunningNormalizer(shape=(), clip=10.0)
        
        # For return normalization
        self.ret = 0.0
    
    def update(self, reward: float, done: bool = False) -> None:
        """
        Update statistics with new reward.
        
        Args:
            reward: New reward value
            done: Whether episode ended
        """
        self.ret = self.ret * self.gamma + reward
        self.normalizer.update(np.array([self.ret]))
        
        if done:
            self.ret = 0.0
    
    def normalize(self, reward: float) -> float:
        """Normalize reward."""
        normalized = self.normalizer.normalize(np.array([reward]))
        return float(normalized[0])


# -----------------------------------------------------------------------------
# Validation Utilities
# -----------------------------------------------------------------------------

def validate_observation(obs: np.ndarray, spec: Optional[ObservationSpec] = None) -> List[str]:
    """
    Validate an observation against the specification.
    
    Args:
        obs: Observation to validate
        spec: Observation specification (uses default if None)
    
    Returns:
        List of validation error messages (empty if valid)
    """
    spec = spec or get_observation_spec()
    errors = []
    
    # Check shape
    if obs.shape != (spec.total_dim,):
        errors.append(f"Shape mismatch: expected ({spec.total_dim},), got {obs.shape}")
        return errors
    
    # Check for NaN/Inf
    if np.any(np.isnan(obs)):
        nan_indices = np.where(np.isnan(obs))[0]
        nan_features = [spec.all_features[i] for i in nan_indices if i < len(spec.all_features)]
        errors.append(f"NaN values in features: {nan_features}")
    
    if np.any(np.isinf(obs)):
        inf_indices = np.where(np.isinf(obs))[0]
        inf_features = [spec.all_features[i] for i in inf_indices if i < len(spec.all_features)]
        errors.append(f"Inf values in features: {inf_features}")
    
    return errors


def validate_action(action: np.ndarray, spec: ActionSpec) -> List[str]:
    """
    Validate an action against the specification.
    
    Args:
        action: Action to validate
        spec: Action specification
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check shape
    if action.shape != (spec.dim,):
        errors.append(f"Shape mismatch: expected ({spec.dim},), got {action.shape}")
        return errors
    
    # Check bounds
    if spec.is_discrete:
        if not np.issubdtype(action.dtype, np.integer):
            errors.append(f"Discrete action should be integer, got {action.dtype}")
        if action[0] < 0 or action[0] >= spec.n_discrete:
            errors.append(f"Action {action[0]} out of range [0, {spec.n_discrete})")
    else:
        if np.any(action < spec.low) or np.any(action > spec.high):
            errors.append(f"Action {action} out of bounds [{spec.low}, {spec.high}]")
    
    return errors

