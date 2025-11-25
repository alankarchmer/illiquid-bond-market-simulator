"""
Multi-agent environment for illiquid bond market making.

This module provides a PettingZoo-compatible multi-agent environment where
multiple dealers compete to provide liquidity in the bond market.

The environment supports:
- Multiple learning agents (dealers) competing for client flow
- Parallel API (all agents act simultaneously)
- Turn-based API (agents take turns responding to RFQs)
- Configurable competition dynamics

PettingZoo API Reference:
- agents: List of agent IDs
- possible_agents: List of all possible agent IDs
- observation_spaces: Dict mapping agent ID to observation space
- action_spaces: Dict mapping agent ID to action space
- reset(): Reset and return observations
- step(actions): Take actions and return observations, rewards, etc.
- observe(agent): Get observation for specific agent
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from illiquid_market_sim.bonds import Bond, generate_bond_universe
from illiquid_market_sim.clients import Client, create_client_universe
from illiquid_market_sim.config import SimulationConfig
from illiquid_market_sim.env import EnvConfig, ObservationSpace, ActionSpace, ActionType
from illiquid_market_sim.market import MarketImpactModel, MarketState
from illiquid_market_sim.portfolio import Portfolio
from illiquid_market_sim.rfq import Quote, RFQ, Trade


class CompetitionMode(str, Enum):
    """How agents compete for RFQs."""
    WINNER_TAKES_ALL = "winner_takes_all"  # Best quote wins
    PROBABILISTIC = "probabilistic"         # Better quotes more likely to win
    AUCTION = "auction"                     # All quotes visible, client picks


@dataclass
class MultiAgentEnvConfig:
    """
    Configuration for multi-agent environment.
    
    Attributes:
        num_agents: Number of dealer agents
        competition_mode: How agents compete for RFQs
        sim_config: Underlying simulation configuration
        max_episode_steps: Maximum steps per episode
        share_market_info: Whether agents can see each other's positions
        client_loyalty: Probability client sticks with previous dealer
    """
    num_agents: int = 2
    competition_mode: CompetitionMode = CompetitionMode.WINNER_TAKES_ALL
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)
    max_episode_steps: int = 100
    share_market_info: bool = False
    client_loyalty: float = 0.0
    action_type: ActionType = ActionType.CONTINUOUS_SPREAD
    seed: Optional[int] = None


class DealerState:
    """State for a single dealer agent."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.portfolio = Portfolio()
        self.trades: List[Trade] = []
        self.rfqs_received: List[RFQ] = []
        self.pnl_history: List[float] = [0.0]
        self.trade_counter = 0
    
    def reset(self) -> None:
        """Reset dealer state."""
        self.portfolio = Portfolio()
        self.trades = []
        self.rfqs_received = []
        self.pnl_history = [0.0]
        self.trade_counter = 0
    
    def get_pnl(self, bonds: List[Bond]) -> float:
        """Get current PnL."""
        mtm = self.portfolio.mark_to_market(bonds)
        return mtm['total_pnl']


class MultiAgentTradingEnv:
    """
    PettingZoo-compatible multi-agent trading environment.
    
    Multiple dealer agents compete to provide liquidity to clients.
    Each agent has their own portfolio and receives their own observations.
    
    The environment follows the Parallel API where all agents act
    simultaneously on each step.
    
    Example:
        >>> config = MultiAgentEnvConfig(num_agents=3)
        >>> env = MultiAgentTradingEnv(config)
        >>> observations, infos = env.reset()
        >>> 
        >>> while env.agents:
        ...     actions = {agent: env.action_space(agent).sample() 
        ...                for agent in env.agents}
        ...     observations, rewards, terminations, truncations, infos = env.step(actions)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "name": "multi_agent_trading"}
    
    def __init__(self, config: Optional[MultiAgentEnvConfig] = None):
        """
        Initialize multi-agent environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config or MultiAgentEnvConfig()
        self._seed = self.config.seed
        
        # Agent setup
        self.possible_agents = [f"dealer_{i}" for i in range(self.config.num_agents)]
        self.agents: List[str] = []
        
        # Spaces
        self._observation_space = ObservationSpace()
        self._action_space = ActionSpace(
            action_type=self.config.action_type,
            n_discrete_levels=10,
            min_spread_bps=10.0,
            max_spread_bps=200.0,
        )
        
        # State
        self._dealer_states: Dict[str, DealerState] = {}
        self._bonds: List[Bond] = []
        self._bonds_dict: Dict[str, Bond] = {}
        self._clients: List[Client] = []
        self._market_state: Optional[MarketState] = None
        self._impact_model: Optional[MarketImpactModel] = None
        
        self._current_step = 0
        self._current_rfq: Optional[RFQ] = None
        self._current_bond: Optional[Bond] = None
        self._current_client: Optional[Client] = None
        self._pending_rfqs: List[Tuple[Client, RFQ]] = []
        
        # Client-dealer relationships (for loyalty)
        self._client_preferred_dealer: Dict[str, str] = {}
        
        self._initialized = False
    
    # -------------------------------------------------------------------------
    # PettingZoo Parallel API
    # -------------------------------------------------------------------------
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Reset environment and return initial observations.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Tuple of (observations dict, infos dict)
        """
        if seed is not None:
            self._seed = seed
        
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
        
        # Reset agents
        self.agents = self.possible_agents.copy()
        
        # Initialize dealer states
        self._dealer_states = {
            agent_id: DealerState(agent_id) for agent_id in self.agents
        }
        
        # Initialize market
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
        )
        
        self._market_state = MarketState(
            volatility=self.config.sim_config.market_volatility,
            jump_probability=self.config.sim_config.jump_probability,
        )
        
        self._impact_model = MarketImpactModel(
            base_impact_coeff=self.config.sim_config.base_impact_coeff,
            cross_impact_factor=self.config.sim_config.cross_impact_factor,
        )
        
        # Reset tracking
        self._current_step = 0
        self._pending_rfqs = []
        self._client_preferred_dealer = {}
        
        # Advance to first RFQ
        self._advance_to_next_rfq()
        
        self._initialized = True
        
        # Get observations for all agents
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}
        
        return observations, infos
    
    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]]
    ]:
        """
        Execute actions for all agents.
        
        Args:
            actions: Dict mapping agent ID to action
        
        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Initialize returns
        observations = {}
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {}
        
        if self._current_rfq is None:
            # No RFQ to process
            self._advance_to_next_rfq()
        else:
            # Process the current RFQ with all agent actions
            step_rewards = self._process_rfq_with_competition(actions)
            for agent, reward in step_rewards.items():
                rewards[agent] = reward
        
        # Advance to next RFQ
        self._advance_to_next_rfq()
        
        # Check truncation
        if self._current_step >= self.config.max_episode_steps:
            truncations = {agent: True for agent in self.agents}
        
        # Get observations and infos
        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
            infos[agent] = self._get_info(agent)
        
        return observations, rewards, terminations, truncations, infos
    
    def observe(self, agent: str) -> np.ndarray:
        """Get observation for a specific agent."""
        return self._get_observation(agent)
    
    def action_space(self, agent: str) -> ActionSpace:
        """Get action space for an agent."""
        return self._action_space
    
    def observation_space(self, agent: str) -> ObservationSpace:
        """Get observation space for an agent."""
        return self._observation_space
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render environment state."""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Step: {self._current_step}/{self.config.max_episode_steps}")
        output.append(f"Active agents: {len(self.agents)}")
        
        for agent_id in self.agents:
            state = self._dealer_states[agent_id]
            pnl = state.get_pnl(self._bonds)
            output.append(f"\n{agent_id}:")
            output.append(f"  PnL: {pnl:+.2f}")
            output.append(f"  Trades: {len(state.trades)}")
            output.append(f"  Inventory: {state.portfolio.get_inventory_risk():.2f}")
        
        if self._current_rfq:
            output.append(f"\nCurrent RFQ: {self._current_rfq}")
        
        output.append(f"{'='*60}\n")
        
        result = "\n".join(output)
        
        if mode == "human":
            print(result)
            return None
        return result
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------
    
    def _advance_to_next_rfq(self) -> None:
        """Advance simulation until we have an RFQ."""
        max_attempts = 10
        
        for _ in range(max_attempts):
            if self._pending_rfqs:
                client, rfq = self._pending_rfqs.pop(0)
                self._current_rfq = rfq
                self._current_bond = self._bonds_dict.get(rfq.bond_id)
                self._current_client = client
                return
            
            self._current_step += 1
            if self._current_step > self.config.max_episode_steps:
                self._current_rfq = None
                return
            
            # Step market
            self._market_state.step(self._bonds)
            
            # Generate RFQs
            for client in self._clients:
                rfq = client.maybe_generate_rfq(
                    timestep=self._current_step,
                    market_state=self._market_state,
                    bonds=self._bonds
                )
                if rfq:
                    self._pending_rfqs.append((client, rfq))
            
            # Update PnL history for all dealers
            for agent_id, state in self._dealer_states.items():
                pnl = state.get_pnl(self._bonds)
                state.pnl_history.append(pnl)
        
        self._current_rfq = None
    
    def _process_rfq_with_competition(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Process RFQ with all agents competing.
        
        Args:
            actions: Actions from all agents
        
        Returns:
            Rewards for each agent
        """
        if self._current_rfq is None or self._current_bond is None:
            return {agent: 0.0 for agent in self.agents}
        
        rfq = self._current_rfq
        bond = self._current_bond
        client = self._current_client
        
        # Get quotes from all agents
        quotes: Dict[str, Quote] = {}
        for agent_id, action in actions.items():
            quote = self._generate_quote(agent_id, rfq, bond, action)
            quotes[agent_id] = quote
        
        # Determine winner based on competition mode
        winning_agent = self._determine_winner(rfq, quotes, client)
        
        # Calculate rewards
        rewards = {agent: 0.0 for agent in self.agents}
        
        if winning_agent is not None:
            # Winner executes trade
            quote = quotes[winning_agent]
            fair_value = client.get_fair_value_estimate(bond, self._market_state)
            will_trade = client.decide_trade(rfq, quote, fair_value)
            
            if will_trade:
                reward = self._execute_trade(winning_agent, rfq, quote, bond)
                rewards[winning_agent] = reward
                
                # Update client preference
                self._client_preferred_dealer[client.client_id] = winning_agent
        
        # Small penalty for losers (opportunity cost)
        for agent_id in self.agents:
            if agent_id != winning_agent:
                rewards[agent_id] = -0.01  # Small penalty for not winning
        
        return rewards
    
    def _generate_quote(
        self,
        agent_id: str,
        rfq: RFQ,
        bond: Bond,
        action: np.ndarray
    ) -> Quote:
        """Generate quote for an agent."""
        state = self._dealer_states[agent_id]
        portfolio = state.portfolio
        
        # Decode action to spread
        action_params = self._action_space.decode_action(action)
        spread_bps = action_params.get("spread_bps", 50.0)
        
        # Get naive mid
        last_price = bond.get_last_traded_price()
        if last_price is not None:
            naive_mid = last_price
        else:
            naive_mid = bond.get_naive_mid(self._market_state.get_factors())
        
        # Adjust for inventory
        position = portfolio.get_position(bond.id)
        inventory = position.quantity
        inventory_adjustment = 0.0
        
        if rfq.side == "buy":
            inventory_adjustment = -inventory * 0.01
        else:
            inventory_adjustment = inventory * 0.01
        
        # Calculate price
        spread_pts = spread_bps / 100.0
        
        if rfq.side == "buy":
            price = naive_mid + spread_pts / 2 + inventory_adjustment
        else:
            price = naive_mid - spread_pts / 2 + inventory_adjustment
        
        return Quote(
            rfq_id=rfq.rfq_id,
            price=round(price, 2),
            spread_bps=round(spread_bps, 1),
            timestamp=rfq.timestamp
        )
    
    def _determine_winner(
        self,
        rfq: RFQ,
        quotes: Dict[str, Quote],
        client: Client
    ) -> Optional[str]:
        """
        Determine which agent wins the RFQ.
        
        Args:
            rfq: The RFQ
            quotes: Quotes from all agents
            client: The client
        
        Returns:
            Winning agent ID, or None if no winner
        """
        if not quotes:
            return None
        
        # Check client loyalty
        if self.config.client_loyalty > 0:
            preferred = self._client_preferred_dealer.get(client.client_id)
            if preferred and preferred in quotes:
                if random.random() < self.config.client_loyalty:
                    return preferred
        
        if self.config.competition_mode == CompetitionMode.WINNER_TAKES_ALL:
            # Best price wins
            if rfq.side == "buy":
                # Client buying - lowest offer wins
                winner = min(quotes.keys(), key=lambda a: quotes[a].price)
            else:
                # Client selling - highest bid wins
                winner = max(quotes.keys(), key=lambda a: quotes[a].price)
            return winner
        
        elif self.config.competition_mode == CompetitionMode.PROBABILISTIC:
            # Better quotes more likely to win
            prices = {a: q.price for a, q in quotes.items()}
            
            if rfq.side == "buy":
                # Lower price = better for client buying
                min_price = min(prices.values())
                max_price = max(prices.values())
                if max_price == min_price:
                    weights = {a: 1.0 for a in quotes}
                else:
                    weights = {a: (max_price - p) / (max_price - min_price + 0.01) 
                              for a, p in prices.items()}
            else:
                # Higher price = better for client selling
                min_price = min(prices.values())
                max_price = max(prices.values())
                if max_price == min_price:
                    weights = {a: 1.0 for a in quotes}
                else:
                    weights = {a: (p - min_price) / (max_price - min_price + 0.01) 
                              for a, p in prices.items()}
            
            # Normalize and sample
            total = sum(weights.values())
            if total == 0:
                return random.choice(list(quotes.keys()))
            
            probs = [weights[a] / total for a in quotes]
            return random.choices(list(quotes.keys()), weights=probs, k=1)[0]
        
        elif self.config.competition_mode == CompetitionMode.AUCTION:
            # Random selection (simplified auction)
            return random.choice(list(quotes.keys()))
        
        return None
    
    def _execute_trade(
        self,
        agent_id: str,
        rfq: RFQ,
        quote: Quote,
        bond: Bond
    ) -> float:
        """Execute trade for winning agent."""
        state = self._dealer_states[agent_id]
        
        # Create trade
        state.trade_counter += 1
        trade_id = f"{agent_id}_T{state.trade_counter:05d}"
        trade = Trade.from_rfq_and_quote(rfq, quote, trade_id)
        state.trades.append(trade)
        
        # Update portfolio
        state.portfolio.update_on_trade(
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
        
        # Calculate reward (edge captured)
        true_fair = bond.get_true_fair_price()
        if rfq.side == "buy":
            edge = quote.price - true_fair
        else:
            edge = true_fair - quote.price
        
        reward = edge * rfq.size
        
        return reward
    
    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for an agent."""
        state = self._dealer_states[agent_id]
        obs = np.zeros(self._observation_space.total_dim, dtype=np.float32)
        
        # RFQ features
        if self._current_rfq:
            rfq = self._current_rfq
            obs[0] = 1.0 if rfq.side == "buy" else -1.0
            obs[1] = rfq.size / 5.0
            
            if self._current_client:
                client_type = self._current_client.client_type
                obs[2] = 1.0 if client_type == "real_money" else 0.0
                obs[3] = 1.0 if client_type == "hedge_fund" else 0.0
                obs[4] = 1.0 if client_type == "fisher" else 0.0
                obs[5] = 1.0 if client_type == "noise" else 0.0
        
        obs[6] = self._current_step / self.config.max_episode_steps
        
        # Bond features
        if self._current_bond:
            bond = self._current_bond
            obs[8] = 1.0 if bond.sector == "IG" else 0.0
            obs[9] = 1.0 if bond.sector == "HY" else 0.0
            obs[10] = 1.0 if bond.sector == "EM" else 0.0
            obs[11] = bond.liquidity
            obs[12] = bond.volatility * 50.0
            obs[13] = bond.maturity_years / 30.0
            
            position = state.portfolio.get_position(bond.id)
            obs[17] = position.quantity / 10.0
        
        # Portfolio features
        pnl = state.get_pnl(self._bonds)
        obs[18] = pnl / 100.0
        obs[19] = state.portfolio.realized_pnl / 100.0
        obs[20] = state.portfolio.get_inventory_risk() / 50.0
        obs[21] = len(state.trades) / 100.0
        
        # Market features
        if self._market_state:
            factors = self._market_state.get_factors()
            obs[24] = factors.get("level_IG", 0.0) / 5.0
            obs[25] = factors.get("level_HY", 0.0) / 5.0
            obs[26] = factors.get("level_EM", 0.0) / 5.0
        
        # Competition features (if sharing info)
        if self.config.share_market_info:
            other_inventories = []
            for other_id, other_state in self._dealer_states.items():
                if other_id != agent_id:
                    other_inventories.append(other_state.portfolio.get_inventory_risk())
            if other_inventories:
                obs[30] = np.mean(other_inventories) / 50.0
        
        return obs
    
    def _get_info(self, agent_id: str) -> Dict[str, Any]:
        """Get info dict for an agent."""
        state = self._dealer_states[agent_id]
        
        return {
            "step": self._current_step,
            "trades": len(state.trades),
            "pnl": state.get_pnl(self._bonds),
            "inventory_risk": state.portfolio.get_inventory_risk(),
        }


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def make_multi_agent_env(
    num_agents: int = 2,
    competition_mode: str = "winner_takes_all",
    **kwargs
) -> MultiAgentTradingEnv:
    """
    Create a multi-agent trading environment.
    
    Args:
        num_agents: Number of dealer agents
        competition_mode: How agents compete ("winner_takes_all", "probabilistic", "auction")
        **kwargs: Additional config options
    
    Returns:
        MultiAgentTradingEnv instance
    """
    mode = CompetitionMode(competition_mode)
    config = MultiAgentEnvConfig(
        num_agents=num_agents,
        competition_mode=mode,
        **kwargs
    )
    return MultiAgentTradingEnv(config)

