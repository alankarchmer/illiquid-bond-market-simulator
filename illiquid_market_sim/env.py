"""
Gymnasium-compatible RL environment for illiquid bond market making.

This module provides a TradingEnv class that wraps the bond market simulation
and exposes it through the standard Gymnasium API (reset, step, render, etc.).

The environment supports:
- Single-agent mode: One learning agent vs scripted clients
- Configurable observation, action, and reward spaces
- Multiple reward functions (PnL-based, risk-adjusted, etc.)
- Reproducible seeding
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, SupportsFloat

import numpy as np

from illiquid_market_sim.bonds import Bond, generate_bond_universe
from illiquid_market_sim.clients import Client, create_client_universe
from illiquid_market_sim.config import SimulationConfig
from illiquid_market_sim.market import MarketImpactModel, MarketState
from illiquid_market_sim.portfolio import Portfolio
from illiquid_market_sim.rfq import Quote, RFQ, Trade


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

class RewardType(str, Enum):
    """Reward function types."""
    PNL = "pnl"                      # Raw PnL change
    RISK_ADJUSTED_PNL = "risk_adjusted_pnl"  # PnL - inventory penalty
    EXECUTION_QUALITY = "execution_quality"  # Edge captured vs fair value
    SHARPE = "sharpe"                # Rolling Sharpe-like reward
    COMPOSITE = "composite"          # Weighted combination


class ActionType(str, Enum):
    """Action space types."""
    CONTINUOUS_SPREAD = "continuous_spread"  # Quote spread as continuous value
    DISCRETE_SPREAD = "discrete_spread"      # Choose from discrete spread tiers
    CONTINUOUS_PRICE = "continuous_price"    # Quote exact price
    ACCEPT_REJECT = "accept_reject"          # Binary accept/reject + spread


@dataclass
class EnvConfig:
    """
    Configuration for the RL environment.
    
    Attributes:
        sim_config: Underlying simulation configuration
        max_episode_steps: Maximum steps per episode
        reward_type: Type of reward function to use
        action_type: Type of action space
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        reward_scale: Scaling factor for rewards
        inventory_penalty: Penalty coefficient for inventory risk
        discrete_spread_levels: Number of discrete spread levels (if using discrete actions)
        max_spread_bps: Maximum spread in basis points for action clipping
        min_spread_bps: Minimum spread in basis points
        observation_window: Number of past steps to include in observation
        include_client_features: Whether to include client-specific features
        include_market_regime: Whether to include market regime indicators
    """
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Episode settings
    max_episode_steps: int = 100
    
    # Reward settings
    reward_type: RewardType = RewardType.RISK_ADJUSTED_PNL
    reward_scale: float = 1.0
    inventory_penalty: float = 0.01
    
    # Action settings
    action_type: ActionType = ActionType.CONTINUOUS_SPREAD
    discrete_spread_levels: int = 10
    max_spread_bps: float = 200.0
    min_spread_bps: float = 10.0
    
    # Observation settings
    normalize_obs: bool = True
    normalize_reward: bool = False
    observation_window: int = 1
    include_client_features: bool = True
    include_market_regime: bool = True
    
    # Reproducibility
    seed: Optional[int] = None


# -----------------------------------------------------------------------------
# Observation and Action Space Helpers
# -----------------------------------------------------------------------------

@dataclass
class ObservationSpace:
    """
    Defines the observation space structure.
    
    The observation is a dictionary with the following keys:
    - rfq: Current RFQ features (bond_id, side, size, client features)
    - bond: Bond features (sector, rating, liquidity, volatility, etc.)
    - portfolio: Portfolio state (position, cash, pnl, inventory risk)
    - market: Market state (factor levels, regime indicators)
    - history: Recent trading history (optional)
    """
    
    # Feature dimensions
    rfq_dim: int = 8
    bond_dim: int = 10
    portfolio_dim: int = 6
    market_dim: int = 6
    history_dim: int = 10
    
    @property
    def total_dim(self) -> int:
        """Total flattened observation dimension."""
        return self.rfq_dim + self.bond_dim + self.portfolio_dim + self.market_dim + self.history_dim
    
    def get_low(self) -> np.ndarray:
        """Get lower bounds for observation space."""
        return np.full(self.total_dim, -np.inf, dtype=np.float32)
    
    def get_high(self) -> np.ndarray:
        """Get upper bounds for observation space."""
        return np.full(self.total_dim, np.inf, dtype=np.float32)


@dataclass
class ActionSpace:
    """
    Defines the action space structure.
    
    For CONTINUOUS_SPREAD:
        Action is a single float in [0, 1] representing spread level.
        Mapped to [min_spread_bps, max_spread_bps].
    
    For DISCRETE_SPREAD:
        Action is an integer in [0, n_levels-1] representing spread tier.
    
    For CONTINUOUS_PRICE:
        Action is a single float representing price offset from naive mid.
    
    For ACCEPT_REJECT:
        Action is a tuple (accept: bool, spread: float).
    """
    
    action_type: ActionType
    n_discrete_levels: int = 10
    min_spread_bps: float = 10.0
    max_spread_bps: float = 200.0
    
    def get_dim(self) -> int:
        """Get action dimension."""
        if self.action_type == ActionType.CONTINUOUS_SPREAD:
            return 1
        elif self.action_type == ActionType.DISCRETE_SPREAD:
            return 1  # Integer action
        elif self.action_type == ActionType.CONTINUOUS_PRICE:
            return 1
        elif self.action_type == ActionType.ACCEPT_REJECT:
            return 2  # (accept, spread)
        return 1
    
    def get_low(self) -> np.ndarray:
        """Get lower bounds for action space."""
        if self.action_type in (ActionType.CONTINUOUS_SPREAD, ActionType.CONTINUOUS_PRICE):
            return np.array([0.0], dtype=np.float32)
        elif self.action_type == ActionType.ACCEPT_REJECT:
            return np.array([0.0, 0.0], dtype=np.float32)
        return np.array([0], dtype=np.int32)
    
    def get_high(self) -> np.ndarray:
        """Get upper bounds for action space."""
        if self.action_type in (ActionType.CONTINUOUS_SPREAD, ActionType.CONTINUOUS_PRICE):
            return np.array([1.0], dtype=np.float32)
        elif self.action_type == ActionType.ACCEPT_REJECT:
            return np.array([1.0, 1.0], dtype=np.float32)
        return np.array([self.n_discrete_levels - 1], dtype=np.int32)
    
    def decode_action(self, action: np.ndarray) -> Dict[str, float]:
        """
        Decode raw action into trading parameters.
        
        Returns:
            Dict with 'spread_bps' and optionally 'accept' keys.
        """
        if self.action_type == ActionType.CONTINUOUS_SPREAD:
            # Map [0, 1] to [min_spread, max_spread]
            spread_bps = self.min_spread_bps + action[0] * (self.max_spread_bps - self.min_spread_bps)
            return {"spread_bps": float(spread_bps), "accept": True}
        
        elif self.action_type == ActionType.DISCRETE_SPREAD:
            # Map discrete level to spread
            level = int(action[0])
            spread_bps = self.min_spread_bps + (level / (self.n_discrete_levels - 1)) * (self.max_spread_bps - self.min_spread_bps)
            return {"spread_bps": float(spread_bps), "accept": True}
        
        elif self.action_type == ActionType.CONTINUOUS_PRICE:
            # Action is price offset (will be interpreted by quoting logic)
            return {"price_offset": float(action[0]), "accept": True}
        
        elif self.action_type == ActionType.ACCEPT_REJECT:
            accept = action[0] > 0.5
            spread_bps = self.min_spread_bps + action[1] * (self.max_spread_bps - self.min_spread_bps)
            return {"spread_bps": float(spread_bps), "accept": bool(accept)}
        
        return {"spread_bps": 50.0, "accept": True}


# -----------------------------------------------------------------------------
# RL Quoting Strategy
# -----------------------------------------------------------------------------

class RLQuotingStrategy:
    """
    Quoting strategy that uses actions from the RL agent.
    
    This strategy receives action parameters from the environment and
    generates quotes accordingly.
    """
    
    def __init__(
        self,
        base_spread_bps: float = 50.0,
        illiquidity_spread_multiplier: float = 2.0,
        inventory_risk_penalty: float = 0.01,
        max_inventory_per_bond: float = 10.0
    ):
        self.base_spread_bps = base_spread_bps
        self.illiquidity_spread_multiplier = illiquidity_spread_multiplier
        self.inventory_risk_penalty = inventory_risk_penalty
        self.max_inventory_per_bond = max_inventory_per_bond
        
        # Action parameters set by environment
        self._action_params: Dict[str, float] = {}
    
    def set_action(self, action_params: Dict[str, float]) -> None:
        """Set action parameters for next quote."""
        self._action_params = action_params
    
    def generate_quote(
        self,
        rfq: RFQ,
        bond: Bond,
        portfolio: Portfolio,
        market_state: MarketState,
        client_stats: Optional[Any] = None
    ) -> Quote:
        """Generate a quote using RL action parameters."""
        
        # Get naive mid estimate
        naive_mid = self._estimate_mid(bond, portfolio, market_state)
        
        # Get spread from action (or use default)
        spread_bps = self._action_params.get("spread_bps", self.base_spread_bps)
        
        # Check if we should accept (for accept/reject action type)
        accept = self._action_params.get("accept", True)
        if not accept:
            # Quote very wide to ensure no trade
            spread_bps = 1000.0
        
        # Adjust for liquidity (RL can learn to override this)
        liquidity_mult = 1.0 + (1.0 - bond.liquidity) * self.illiquidity_spread_multiplier * 0.5
        effective_spread = spread_bps * liquidity_mult
        
        # Adjust for inventory (RL can learn to manage this)
        position = portfolio.get_position(bond.id)
        inventory = position.quantity
        inventory_adjustment = 0.0
        
        if rfq.side == "buy":
            # Client buying, we're selling
            inventory_adjustment = -inventory * self.inventory_risk_penalty
        else:
            # Client selling, we're buying
            inventory_adjustment = inventory * self.inventory_risk_penalty
        
        # Calculate final quote price
        spread_pts = effective_spread / 100.0
        
        if rfq.side == "buy":
            # Client buying, we're selling -> offer side
            price = naive_mid + spread_pts / 2 + inventory_adjustment
        else:
            # Client selling, we're buying -> bid side
            price = naive_mid - spread_pts / 2 + inventory_adjustment
        
        # Extreme inventory protection
        if abs(inventory) > self.max_inventory_per_bond:
            if rfq.side == "buy" and inventory < 0:
                price += 5.0
            elif rfq.side == "sell" and inventory > 0:
                price -= 5.0
        
        return Quote(
            rfq_id=rfq.rfq_id,
            price=round(price, 2),
            spread_bps=round(effective_spread, 1),
            timestamp=rfq.timestamp
        )
    
    def _estimate_mid(
        self,
        bond: Bond,
        portfolio: Portfolio,
        market_state: MarketState
    ) -> float:
        """Estimate mid price for a bond."""
        last_price = bond.get_last_traded_price()
        if last_price is not None:
            return last_price
        return bond.get_naive_mid(market_state.get_factors())


# -----------------------------------------------------------------------------
# Normalization Utilities
# -----------------------------------------------------------------------------

class RunningMeanStd:
    """Running mean and standard deviation for normalization."""
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        
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
        """Normalize input using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


# -----------------------------------------------------------------------------
# Main Environment Class
# -----------------------------------------------------------------------------

class TradingEnv:
    """
    Gymnasium-compatible RL environment for bond market making.
    
    The agent acts as a dealer, receiving RFQs from clients and deciding
    how to quote. The goal is to maximize risk-adjusted PnL while managing
    inventory and client relationships.
    
    Observation:
        A flattened numpy array containing:
        - RFQ features (side, size, client type indicators)
        - Bond features (sector, rating, liquidity, volatility, etc.)
        - Portfolio state (position, cash, PnL, inventory risk)
        - Market state (factor levels, regime indicators)
    
    Action:
        Depends on action_type configuration:
        - CONTINUOUS_SPREAD: Float in [0, 1] mapped to spread range
        - DISCRETE_SPREAD: Integer selecting spread tier
        - CONTINUOUS_PRICE: Float representing price offset
        - ACCEPT_REJECT: Tuple of (accept decision, spread)
    
    Reward:
        Depends on reward_type configuration:
        - PNL: Raw PnL change from previous step
        - RISK_ADJUSTED_PNL: PnL minus inventory penalty
        - EXECUTION_QUALITY: Edge captured vs fair value
        - SHARPE: Rolling Sharpe-like reward
        - COMPOSITE: Weighted combination of above
    
    Episode Termination:
        - max_episode_steps reached
        - Optional: Inventory limits breached, large drawdown, etc.
    
    Example:
        >>> from illiquid_market_sim.env import TradingEnv, EnvConfig
        >>> config = EnvConfig(max_episode_steps=100)
        >>> env = TradingEnv(config)
        >>> obs, info = env.reset(seed=42)
        >>> for _ in range(100):
        ...     action = env.action_space_sample()  # or use your policy
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     if terminated or truncated:
        ...         break
    """
    
    # Gymnasium metadata
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, config: Optional[EnvConfig] = None):
        """
        Initialize the trading environment.
        
        Args:
            config: Environment configuration. Uses defaults if None.
        """
        self.config = config or EnvConfig()
        self._seed = self.config.seed
        
        # Initialize spaces
        self.observation_space_def = ObservationSpace()
        self.action_space_def = ActionSpace(
            action_type=self.config.action_type,
            n_discrete_levels=self.config.discrete_spread_levels,
            min_spread_bps=self.config.min_spread_bps,
            max_spread_bps=self.config.max_spread_bps
        )
        
        # Normalization
        self._obs_rms = RunningMeanStd(shape=(self.observation_space_def.total_dim,))
        self._reward_rms = RunningMeanStd(shape=())
        
        # State variables (initialized in reset)
        self._bonds: List[Bond] = []
        self._clients: List[Client] = []
        self._market_state: Optional[MarketState] = None
        self._impact_model: Optional[MarketImpactModel] = None
        self._portfolio: Optional[Portfolio] = None
        self._quoting_strategy: Optional[RLQuotingStrategy] = None
        
        self._current_step: int = 0
        self._current_rfq: Optional[RFQ] = None
        self._current_bond: Optional[Bond] = None
        self._pending_rfqs: List[Tuple[Client, RFQ]] = []
        
        # Tracking
        self._all_trades: List[Trade] = []
        self._all_rfqs: List[RFQ] = []
        self._pnl_history: List[float] = []
        self._trade_counter: int = 0
        self._bonds_dict: Dict[str, Bond] = {}
        
        # Episode state
        self._episode_reward: float = 0.0
        self._last_pnl: float = 0.0
        
        # Initialize
        self._initialized = False
    
    # -------------------------------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------------------------------
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused currently)
        
        Returns:
            Tuple of (observation, info dict)
        """
        if seed is not None:
            self._seed = seed
        
        # Set seeds
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            self.config.sim_config.random_seed = self._seed
        
        # Initialize simulation components
        self._bonds = generate_bond_universe(
            self.config.sim_config.num_bonds,
            seed=self._seed
        )
        self._bonds_dict = {b.id: b for b in self._bonds}
        
        self._clients = create_client_universe(
            num_real_money=self.config.sim_config.num_real_money_clients,
            num_hedge_fund=self.config.sim_config.num_hedge_fund_clients,
            num_fisher=self.config.sim_config.num_fisher_clients,
            num_noise=self.config.sim_config.num_noise_clients,
            rfq_prob_per_client=self.config.sim_config.rfq_prob_per_client
        )
        
        self._market_state = MarketState(
            volatility=self.config.sim_config.market_volatility,
            jump_probability=self.config.sim_config.jump_probability
        )
        
        self._impact_model = MarketImpactModel(
            base_impact_coeff=self.config.sim_config.base_impact_coeff,
            cross_impact_factor=self.config.sim_config.cross_impact_factor,
            impact_decay=self.config.sim_config.impact_decay
        )
        
        self._portfolio = Portfolio()
        
        self._quoting_strategy = RLQuotingStrategy(
            base_spread_bps=self.config.sim_config.base_spread_bps,
            illiquidity_spread_multiplier=self.config.sim_config.illiquidity_spread_multiplier,
            inventory_risk_penalty=self.config.sim_config.inventory_risk_penalty,
            max_inventory_per_bond=self.config.sim_config.max_inventory_per_bond
        )
        
        # Reset tracking
        self._current_step = 0
        self._all_trades = []
        self._all_rfqs = []
        self._pnl_history = [0.0]
        self._trade_counter = 0
        self._episode_reward = 0.0
        self._last_pnl = 0.0
        self._pending_rfqs = []
        
        # Generate first RFQ
        self._advance_to_next_rfq()
        
        self._initialized = True
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action from the agent
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self._current_rfq is None:
            # No RFQ to process, advance time and try again
            self._advance_to_next_rfq()
            if self._current_rfq is None:
                # Still no RFQ, return zero reward
                obs = self._get_observation()
                reward = 0.0
                terminated = self._current_step >= self.config.max_episode_steps
                truncated = False
                info = self._get_info()
                return obs, reward, terminated, truncated, info
        
        # Decode action
        action_params = self.action_space_def.decode_action(action)
        
        # Set action in quoting strategy
        self._quoting_strategy.set_action(action_params)
        
        # Process the current RFQ
        reward = self._process_current_rfq()
        
        # Advance to next RFQ
        self._advance_to_next_rfq()
        
        # Check termination
        terminated = False
        truncated = self._current_step >= self.config.max_episode_steps
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        # Track episode reward
        self._episode_reward += reward
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the environment state.
        
        Args:
            mode: Render mode ("human" or "ansi")
        
        Returns:
            String representation if mode is "ansi", None otherwise
        """
        mtm = self._portfolio.mark_to_market(self._bonds)
        
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Step: {self._current_step}/{self.config.max_episode_steps}")
        output.append(f"Total PnL: {mtm['total_pnl']:+.2f}")
        output.append(f"Inventory Risk: {self._portfolio.get_inventory_risk():.2f}")
        output.append(f"Trades: {len(self._all_trades)}")
        output.append(f"RFQs: {len(self._all_rfqs)}")
        
        if self._current_rfq:
            output.append(f"\nCurrent RFQ: {self._current_rfq}")
            if self._current_bond:
                output.append(f"Bond: {self._current_bond.id} ({self._current_bond.sector}, {self._current_bond.rating})")
        
        output.append(f"{'='*60}\n")
        
        result = "\n".join(output)
        
        if mode == "human":
            print(result)
            return None
        return result
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set random seed (deprecated in Gymnasium, use reset(seed=...) instead).
        
        Args:
            seed: Random seed
        
        Returns:
            List containing the seed
        """
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed] if seed is not None else []
    
    # -------------------------------------------------------------------------
    # Space sampling (for compatibility without gymnasium dependency)
    # -------------------------------------------------------------------------
    
    def action_space_sample(self) -> np.ndarray:
        """Sample a random action from the action space."""
        low = self.action_space_def.get_low()
        high = self.action_space_def.get_high()
        
        if self.config.action_type == ActionType.DISCRETE_SPREAD:
            return np.array([random.randint(0, self.config.discrete_spread_levels - 1)], dtype=np.int32)
        
        return np.random.uniform(low, high).astype(np.float32)
    
    def observation_space_sample(self) -> np.ndarray:
        """Sample a random observation (for testing)."""
        return np.random.randn(self.observation_space_def.total_dim).astype(np.float32)
    
    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------
    
    def _advance_to_next_rfq(self) -> None:
        """
        Advance simulation until we have an RFQ to process.
        
        This may advance multiple time steps if no RFQs are generated.
        """
        max_attempts = 10  # Prevent infinite loop
        
        for _ in range(max_attempts):
            if self._pending_rfqs:
                # Process next pending RFQ
                client, rfq = self._pending_rfqs.pop(0)
                self._current_rfq = rfq
                self._current_bond = self._bonds_dict.get(rfq.bond_id)
                self._current_client = client
                return
            
            # Advance market state
            self._current_step += 1
            if self._current_step > self.config.max_episode_steps:
                self._current_rfq = None
                self._current_bond = None
                return
            
            # Step market
            self._market_state.step(self._bonds)
            
            # Generate RFQs from clients
            for client in self._clients:
                rfq = client.maybe_generate_rfq(
                    timestep=self._current_step,
                    market_state=self._market_state,
                    bonds=self._bonds
                )
                if rfq:
                    self._pending_rfqs.append((client, rfq))
                    self._all_rfqs.append(rfq)
            
            # Mark to market
            mtm = self._portfolio.mark_to_market(self._bonds)
            self._pnl_history.append(mtm['total_pnl'])
        
        # No RFQ found after max attempts
        self._current_rfq = None
        self._current_bond = None
    
    def _process_current_rfq(self) -> float:
        """
        Process the current RFQ and return reward.
        
        Returns:
            Reward for this action
        """
        if self._current_rfq is None or self._current_bond is None:
            return 0.0
        
        rfq = self._current_rfq
        bond = self._current_bond
        client = self._current_client
        
        # Generate quote using RL strategy
        quote = self._quoting_strategy.generate_quote(
            rfq=rfq,
            bond=bond,
            portfolio=self._portfolio,
            market_state=self._market_state
        )
        
        # Client decides whether to trade
        fair_value_estimate = client.get_fair_value_estimate(bond, self._market_state)
        will_trade = client.decide_trade(rfq, quote, fair_value_estimate)
        
        # Calculate reward before trade
        old_pnl = self._pnl_history[-1] if self._pnl_history else 0.0
        
        if will_trade:
            # Execute trade
            self._execute_trade(client, rfq, quote, bond)
        
        # Mark to market and calculate new PnL
        mtm = self._portfolio.mark_to_market(self._bonds)
        new_pnl = mtm['total_pnl']
        
        # Calculate reward
        reward = self._calculate_reward(old_pnl, new_pnl, will_trade, rfq, quote, bond)
        
        # Clear current RFQ
        self._current_rfq = None
        self._current_bond = None
        
        return reward
    
    def _execute_trade(
        self,
        client: Client,
        rfq: RFQ,
        quote: Quote,
        bond: Bond
    ) -> None:
        """Execute a trade."""
        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:05d}"
        
        # Create trade
        trade = Trade.from_rfq_and_quote(rfq, quote, trade_id)
        self._all_trades.append(trade)
        
        # Update portfolio
        self._portfolio.update_on_trade(
            bond_id=trade.bond_id,
            side=trade.side,
            size=trade.size,
            price=trade.price
        )
        
        # Record trade in bond
        bond.record_trade(quote.price)
        
        # Apply market impact
        self._impact_model.apply_trade_impact(
            traded_bond=bond,
            side=trade.side,
            size=trade.size,
            all_bonds=self._bonds
        )
    
    def _calculate_reward(
        self,
        old_pnl: float,
        new_pnl: float,
        traded: bool,
        rfq: RFQ,
        quote: Quote,
        bond: Bond
    ) -> float:
        """
        Calculate reward based on configured reward type.
        
        Args:
            old_pnl: PnL before action
            new_pnl: PnL after action
            traded: Whether a trade occurred
            rfq: The RFQ that was processed
            quote: The quote that was generated
            bond: The bond involved
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        if self.config.reward_type == RewardType.PNL:
            # Simple PnL change
            reward = new_pnl - old_pnl
        
        elif self.config.reward_type == RewardType.RISK_ADJUSTED_PNL:
            # PnL change minus inventory penalty
            pnl_change = new_pnl - old_pnl
            inventory_risk = self._portfolio.get_inventory_risk()
            reward = pnl_change - self.config.inventory_penalty * inventory_risk
        
        elif self.config.reward_type == RewardType.EXECUTION_QUALITY:
            # Edge captured vs fair value
            if traded:
                true_fair = bond.get_true_fair_price()
                if rfq.side == "buy":
                    # We sold - edge is how much above fair we sold
                    edge = quote.price - true_fair
                else:
                    # We bought - edge is how much below fair we bought
                    edge = true_fair - quote.price
                reward = edge * rfq.size
            else:
                reward = 0.0
        
        elif self.config.reward_type == RewardType.SHARPE:
            # Rolling Sharpe-like reward
            pnl_change = new_pnl - old_pnl
            if len(self._pnl_history) > 10:
                recent_returns = np.diff(self._pnl_history[-10:])
                std = np.std(recent_returns) + 1e-8
                reward = pnl_change / std
            else:
                reward = pnl_change
        
        elif self.config.reward_type == RewardType.COMPOSITE:
            # Weighted combination
            pnl_change = new_pnl - old_pnl
            inventory_risk = self._portfolio.get_inventory_risk()
            
            # Edge component
            edge = 0.0
            if traded:
                true_fair = bond.get_true_fair_price()
                if rfq.side == "buy":
                    edge = quote.price - true_fair
                else:
                    edge = true_fair - quote.price
            
            reward = (
                0.5 * pnl_change +
                0.3 * edge * rfq.size -
                0.2 * self.config.inventory_penalty * inventory_risk
            )
        
        # Apply scaling
        reward *= self.config.reward_scale
        
        # Normalize if configured
        if self.config.normalize_reward:
            self._reward_rms.update(np.array([reward]))
            reward = float(self._reward_rms.normalize(np.array([reward]))[0])
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Build observation array from current state.
        
        Returns:
            Flattened observation array
        """
        obs_parts = []
        
        # RFQ features
        rfq_features = self._get_rfq_features()
        obs_parts.append(rfq_features)
        
        # Bond features
        bond_features = self._get_bond_features()
        obs_parts.append(bond_features)
        
        # Portfolio features
        portfolio_features = self._get_portfolio_features()
        obs_parts.append(portfolio_features)
        
        # Market features
        market_features = self._get_market_features()
        obs_parts.append(market_features)
        
        # History features
        history_features = self._get_history_features()
        obs_parts.append(history_features)
        
        # Concatenate all features
        obs = np.concatenate(obs_parts).astype(np.float32)
        
        # Normalize if configured
        if self.config.normalize_obs:
            self._obs_rms.update(obs.reshape(1, -1))
            obs = self._obs_rms.normalize(obs).astype(np.float32)
        
        return obs
    
    def _get_rfq_features(self) -> np.ndarray:
        """Extract features from current RFQ."""
        features = np.zeros(self.observation_space_def.rfq_dim, dtype=np.float32)
        
        if self._current_rfq is None:
            return features
        
        rfq = self._current_rfq
        
        # Side (1 for buy, -1 for sell)
        features[0] = 1.0 if rfq.side == "buy" else -1.0
        
        # Size (normalized)
        features[1] = rfq.size / 5.0  # Normalize by typical max size
        
        # Client type indicators (one-hot style)
        if hasattr(self, '_current_client') and self._current_client is not None:
            client_type = self._current_client.client_type
            features[2] = 1.0 if client_type == "real_money" else 0.0
            features[3] = 1.0 if client_type == "hedge_fund" else 0.0
            features[4] = 1.0 if client_type == "fisher" else 0.0
            features[5] = 1.0 if client_type == "noise" else 0.0
        
        # Timestamp (normalized)
        features[6] = self._current_step / self.config.max_episode_steps
        
        # RFQ count for this client (if available)
        features[7] = min(len(self._all_rfqs) / 100.0, 1.0)
        
        return features
    
    def _get_bond_features(self) -> np.ndarray:
        """Extract features from current bond."""
        features = np.zeros(self.observation_space_def.bond_dim, dtype=np.float32)
        
        if self._current_bond is None:
            return features
        
        bond = self._current_bond
        
        # Sector one-hot
        features[0] = 1.0 if bond.sector == "IG" else 0.0
        features[1] = 1.0 if bond.sector == "HY" else 0.0
        features[2] = 1.0 if bond.sector == "EM" else 0.0
        
        # Liquidity
        features[3] = bond.liquidity
        
        # Volatility (normalized)
        features[4] = bond.volatility * 50.0  # Scale up for visibility
        
        # Maturity (normalized)
        features[5] = bond.maturity_years / 30.0
        
        # Base spread (normalized)
        features[6] = bond.base_spread / 1000.0
        
        # Last traded price (if available, normalized around 100)
        last_price = bond.get_last_traded_price()
        if last_price is not None:
            features[7] = (last_price - 100.0) / 10.0
        
        # Naive mid estimate
        if self._market_state:
            naive_mid = bond.get_naive_mid(self._market_state.get_factors())
            features[8] = (naive_mid - 100.0) / 10.0
        
        # Current position in this bond
        if self._portfolio:
            position = self._portfolio.get_position(bond.id)
            features[9] = position.quantity / 10.0  # Normalize
        
        return features
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Extract portfolio features."""
        features = np.zeros(self.observation_space_def.portfolio_dim, dtype=np.float32)
        
        if self._portfolio is None:
            return features
        
        # Total PnL (normalized)
        if self._pnl_history:
            features[0] = self._pnl_history[-1] / 100.0
        
        # Realized PnL
        features[1] = self._portfolio.realized_pnl / 100.0
        
        # Inventory risk
        features[2] = self._portfolio.get_inventory_risk() / 50.0
        
        # Number of positions
        num_positions = sum(1 for p in self._portfolio.positions.values() if abs(p.quantity) > 0.01)
        features[3] = num_positions / self.config.sim_config.num_bonds
        
        # Total trades
        features[4] = len(self._all_trades) / 100.0
        
        # Fill ratio
        if self._all_rfqs:
            features[5] = len(self._all_trades) / len(self._all_rfqs)
        
        return features
    
    def _get_market_features(self) -> np.ndarray:
        """Extract market state features."""
        features = np.zeros(self.observation_space_def.market_dim, dtype=np.float32)
        
        if self._market_state is None:
            return features
        
        factors = self._market_state.get_factors()
        
        # Factor levels (normalized)
        features[0] = factors.get("level_IG", 0.0) / 5.0
        features[1] = factors.get("level_HY", 0.0) / 5.0
        features[2] = factors.get("level_EM", 0.0) / 5.0
        
        # Time in episode
        features[3] = self._current_step / self.config.max_episode_steps
        
        # Recent volatility estimate (from PnL history)
        if len(self._pnl_history) > 5:
            recent_returns = np.diff(self._pnl_history[-5:])
            features[4] = np.std(recent_returns)
        
        # Market regime indicator (simple heuristic)
        # High absolute factor levels = stressed market
        total_factor_abs = sum(abs(v) for v in factors.values())
        features[5] = min(total_factor_abs / 10.0, 1.0)
        
        return features
    
    def _get_history_features(self) -> np.ndarray:
        """Extract recent history features."""
        features = np.zeros(self.observation_space_def.history_dim, dtype=np.float32)
        
        # Recent PnL trajectory
        if len(self._pnl_history) >= 5:
            recent_pnl = self._pnl_history[-5:]
            for i, pnl in enumerate(recent_pnl):
                features[i] = pnl / 100.0
        
        # Recent trade info
        if self._all_trades:
            last_trade = self._all_trades[-1]
            features[5] = 1.0 if last_trade.side == "buy" else -1.0
            features[6] = last_trade.size / 5.0
            features[7] = (last_trade.price - 100.0) / 10.0
        
        # Episode progress
        features[8] = self._current_step / self.config.max_episode_steps
        
        # Cumulative episode reward (normalized)
        features[9] = self._episode_reward / 100.0
        
        return features
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary."""
        info = {
            "step": self._current_step,
            "total_trades": len(self._all_trades),
            "total_rfqs": len(self._all_rfqs),
            "episode_reward": self._episode_reward,
        }
        
        if self._portfolio and self._bonds:
            mtm = self._portfolio.mark_to_market(self._bonds)
            info["total_pnl"] = mtm["total_pnl"]
            info["realized_pnl"] = mtm["realized_pnl"]
            info["unrealized_pnl"] = mtm["unrealized_pnl"]
            info["inventory_risk"] = self._portfolio.get_inventory_risk()
        
        if self._all_rfqs:
            info["fill_ratio"] = len(self._all_trades) / len(self._all_rfqs)
        
        return info
    
    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current environment state."""
        return {
            "step": self._current_step,
            "max_steps": self.config.max_episode_steps,
            "total_trades": len(self._all_trades),
            "total_rfqs": len(self._all_rfqs),
            "current_pnl": self._pnl_history[-1] if self._pnl_history else 0.0,
            "inventory_risk": self._portfolio.get_inventory_risk() if self._portfolio else 0.0,
            "episode_reward": self._episode_reward,
        }
    
    def get_pnl_history(self) -> List[float]:
        """Get PnL history for the current episode."""
        return self._pnl_history.copy()
    
    def get_trades(self) -> List[Trade]:
        """Get all trades from current episode."""
        return self._all_trades.copy()


# -----------------------------------------------------------------------------
# Gymnasium wrapper (optional, for full compatibility)
# -----------------------------------------------------------------------------

def make_gymnasium_env(config: Optional[EnvConfig] = None) -> "TradingEnv":
    """
    Create a TradingEnv with Gymnasium-compatible spaces.
    
    This function attempts to import gymnasium and set up proper
    space objects. Falls back to the base TradingEnv if gymnasium
    is not available.
    
    Args:
        config: Environment configuration
    
    Returns:
        TradingEnv instance (with gymnasium spaces if available)
    """
    env = TradingEnv(config)
    
    try:
        import gymnasium as gym
        from gymnasium import spaces
        
        # Set up proper gymnasium spaces
        obs_dim = env.observation_space_def.total_dim
        env.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        if env.config.action_type == ActionType.DISCRETE_SPREAD:
            env.action_space = spaces.Discrete(env.config.discrete_spread_levels)
        elif env.config.action_type == ActionType.ACCEPT_REJECT:
            env.action_space = spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        else:
            env.action_space = spaces.Box(
                low=np.array([0.0]),
                high=np.array([1.0]),
                dtype=np.float32
            )
        
    except ImportError:
        pass  # Gymnasium not available, use basic spaces
    
    return env

