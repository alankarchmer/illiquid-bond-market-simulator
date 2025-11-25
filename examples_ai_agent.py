#!/usr/bin/env python3
"""
Examples of plugging AI agents into the illiquid bond market simulator.

This demonstrates how to:
1. Create custom quoting strategies
2. Use historical data for learning
3. Structure for RL agents
4. Structure for LLM-based agents
"""

from typing import Optional
import random

from illiquid_market_sim.simulation import Simulator
from illiquid_market_sim.config import SimulationConfig
from illiquid_market_sim.rfq import Quote
from illiquid_market_sim.bonds import Bond
from illiquid_market_sim.portfolio import Portfolio
from illiquid_market_sim.market import MarketState
from illiquid_market_sim.clients import ClientStats


# ============================================================================
# Example 1: Simple ML-Ready Strategy
# ============================================================================

class MLQuotingStrategy:
    """
    Example of an ML-ready quoting strategy.
    
    This shows the structure you'd use to plug in a trained model.
    Currently uses dummy logic, but you'd replace with your model.
    """
    
    def __init__(self, model=None):
        """
        Args:
            model: Your trained model (sklearn, pytorch, etc.)
        """
        self.model = model
        self.base_spread = 50.0
    
    def generate_quote(
        self,
        rfq,
        bond: Bond,
        portfolio: Portfolio,
        market_state: MarketState,
        client_stats: Optional[ClientStats]
    ) -> Quote:
        """Generate quote using ML model."""
        
        # Extract features
        features = self._extract_features(rfq, bond, portfolio, market_state, client_stats)
        
        # Predict spread adjustment using model
        # In real implementation: spread_adjustment = self.model.predict(features)
        spread_adjustment = self._dummy_predict(features)
        
        # Calculate quote
        naive_mid = self._estimate_mid(bond, market_state)
        spread_bps = self.base_spread * (1 + spread_adjustment)
        
        if rfq.side == "buy":
            price = naive_mid + spread_bps / 100 / 2
        else:
            price = naive_mid - spread_bps / 100 / 2
        
        return Quote(
            rfq_id=rfq.rfq_id,
            price=round(price, 2),
            spread_bps=round(spread_bps, 1),
            timestamp=rfq.timestamp
        )
    
    def _extract_features(self, rfq, bond, portfolio, market_state, client_stats):
        """Extract features for ML model."""
        position = portfolio.get_position(bond.id)
        
        return {
            # Bond features
            'liquidity': bond.liquidity,
            'volatility': bond.volatility,
            'maturity': bond.maturity_years,
            'sector_IG': 1 if bond.sector == 'IG' else 0,
            'sector_HY': 1 if bond.sector == 'HY' else 0,
            'sector_EM': 1 if bond.sector == 'EM' else 0,
            
            # RFQ features
            'size': rfq.size,
            'side_buy': 1 if rfq.side == 'buy' else 0,
            
            # Portfolio features
            'position': position.quantity,
            'total_inventory': portfolio.get_inventory_risk(),
            
            # Client features
            'client_rfq_count': client_stats.rfq_count if client_stats else 0,
            'client_fill_ratio': client_stats.get_fill_ratio() if client_stats else 0,
            'client_avg_edge': client_stats.get_avg_edge() if client_stats else 0,
            
            # Market features
            'factor_IG': market_state.level_IG,
            'factor_HY': market_state.level_HY,
            'factor_EM': market_state.level_EM,
        }
    
    def _dummy_predict(self, features):
        """Dummy prediction - replace with your model."""
        # Example: widen for illiquid bonds, high inventory, low client fill ratio
        adjustment = 0.0
        adjustment += (1 - features['liquidity']) * 0.5
        adjustment += abs(features['position']) / 10 * 0.3
        if features['client_fill_ratio'] < 0.3:
            adjustment += 0.5
        return adjustment
    
    def _estimate_mid(self, bond, market_state):
        """Estimate mid price."""
        last = bond.get_last_traded_price()
        if last:
            return last
        return bond.get_naive_mid(market_state.get_factors())


# ============================================================================
# Example 2: Reinforcement Learning Agent Structure
# ============================================================================

class RLQuotingAgent:
    """
    Template for an RL-based quoting agent.
    
    State: Current market conditions, position, client info
    Action: Spread adjustment to apply
    Reward: Realized P&L + inventory penalty
    """
    
    def __init__(self, state_dim=20, action_dim=5):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions (spread levels)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.policy_network = YourNeuralNetwork(state_dim, action_dim)
        
        # Action space: spread multipliers
        self.actions = [0.5, 0.75, 1.0, 1.5, 2.0]  # Spread multipliers
        
    def generate_quote(self, rfq, bond, portfolio, market_state, client_stats) -> Quote:
        """Generate quote using RL policy."""
        
        # Get state
        state = self._get_state(rfq, bond, portfolio, market_state, client_stats)
        
        # Select action (spread multiplier)
        # action_idx = self.policy_network.select_action(state)
        action_idx = self._dummy_select_action(state)
        spread_multiplier = self.actions[action_idx]
        
        # Calculate quote
        base_spread = 50.0
        spread_bps = base_spread * spread_multiplier
        
        naive_mid = bond.get_last_traded_price() or 100.0
        
        if rfq.side == "buy":
            price = naive_mid + spread_bps / 100 / 2
        else:
            price = naive_mid - spread_bps / 100 / 2
        
        return Quote(
            rfq_id=rfq.rfq_id,
            price=round(price, 2),
            spread_bps=round(spread_bps, 1),
            timestamp=rfq.timestamp
        )
    
    def _get_state(self, rfq, bond, portfolio, market_state, client_stats):
        """Convert current situation to state vector."""
        position = portfolio.get_position(bond.id)
        
        # Normalize features
        state = [
            bond.liquidity,
            bond.volatility * 100,
            bond.maturity_years / 30,
            rfq.size / 5,
            1 if rfq.side == 'buy' else -1,
            position.quantity / 10,
            portfolio.get_inventory_risk() / 50,
            client_stats.get_fill_ratio() if client_stats else 0.5,
            market_state.level_IG / 10,
            market_state.level_HY / 10,
        ]
        
        # Pad to state_dim
        state += [0.0] * (self.state_dim - len(state))
        return state[:self.state_dim]
    
    def _dummy_select_action(self, state):
        """Dummy action selection - replace with your policy."""
        # Random for now
        return random.randint(0, self.action_dim - 1)
    
    def calculate_reward(self, trade_pnl, inventory_change, fill):
        """
        Calculate reward for RL training.
        
        Args:
            trade_pnl: P&L from this trade
            inventory_change: Change in inventory risk
            fill: Whether trade filled
        """
        reward = 0.0
        
        # Reward for P&L
        reward += trade_pnl
        
        # Penalty for inventory
        reward -= abs(inventory_change) * 0.1
        
        # Small reward for fills (want some activity)
        if fill:
            reward += 0.1
        
        return reward


# ============================================================================
# Example 3: LLM-Based Agent Structure
# ============================================================================

class LLMQuotingAgent:
    """
    Template for an LLM-based quoting agent.
    
    Uses LLM to reason about the quote based on context.
    """
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Your LLM API client (OpenAI, Anthropic, etc.)
        """
        self.llm_client = llm_client
        self.base_spread = 50.0
    
    def generate_quote(self, rfq, bond, portfolio, market_state, client_stats) -> Quote:
        """Generate quote using LLM reasoning."""
        
        # Build context for LLM
        context = self._build_context(rfq, bond, portfolio, market_state, client_stats)
        
        # Query LLM for spread decision
        # response = self.llm_client.query(context)
        # spread_multiplier = self._parse_llm_response(response)
        spread_multiplier = self._dummy_llm_decision(context)
        
        # Calculate quote
        spread_bps = self.base_spread * spread_multiplier
        naive_mid = bond.get_last_traded_price() or 100.0
        
        if rfq.side == "buy":
            price = naive_mid + spread_bps / 100 / 2
        else:
            price = naive_mid - spread_bps / 100 / 2
        
        return Quote(
            rfq_id=rfq.rfq_id,
            price=round(price, 2),
            spread_bps=round(spread_bps, 1),
            timestamp=rfq.timestamp
        )
    
    def _build_context(self, rfq, bond, portfolio, market_state, client_stats):
        """Build natural language context for LLM."""
        position = portfolio.get_position(bond.id)
        
        context = f"""
You are a bond dealer. A client wants to {rfq.side} {rfq.size:.1f} units of a bond.

Bond Details:
- Sector: {bond.sector}, Rating: {bond.rating}
- Maturity: {bond.maturity_years:.1f} years
- Liquidity: {bond.liquidity:.2f} (0=illiquid, 1=liquid)
- Volatility: {bond.volatility:.3f}

Your Position:
- Current position in this bond: {position.quantity:.2f}
- Total inventory risk: {portfolio.get_inventory_risk():.1f}

Client History:
"""
        if client_stats and client_stats.rfq_count > 0:
            context += f"""- RFQs: {client_stats.rfq_count}, Trades: {client_stats.trade_count}
- Fill ratio: {client_stats.get_fill_ratio():.1%}
- Average edge: {client_stats.get_avg_edge():+.2f}
"""
        else:
            context += "- New client, no history\n"
        
        context += """
Market Conditions:
- Generally stable with normal volatility

What spread multiplier should you use? (1.0 = base, >1 = wider, <1 = tighter)
Consider: liquidity, position, client toxicity, market conditions.
"""
        return context
    
    def _dummy_llm_decision(self, context):
        """Dummy LLM decision - replace with actual LLM call."""
        # Simple heuristic for demo
        if 'illiquid' in context.lower() or 'Liquidity: 0.' in context:
            return 1.5
        if 'Fill ratio: 0%' in context or 'Fill ratio: 1' in context and '0%' in context:
            return 1.3
        return 1.0


# ============================================================================
# Run Examples
# ============================================================================

def example_ml_strategy():
    """Example: Run simulation with ML strategy."""
    print("Running simulation with ML-based strategy...")
    
    config = SimulationConfig(num_bonds=20, num_steps=30, random_seed=42)
    strategy = MLQuotingStrategy()
    
    sim = Simulator(config=config, custom_quoting_strategy=strategy)
    result = sim.run(verbose=False)
    
    print(f"Result: {result.total_trades} trades, Final P&L: {result.final_pnl:+.2f}")
    print()


def example_rl_agent():
    """Example: Run simulation with RL agent."""
    print("Running simulation with RL agent...")
    
    config = SimulationConfig(num_bonds=20, num_steps=30, random_seed=42)
    agent = RLQuotingAgent()
    
    sim = Simulator(config=config, custom_quoting_strategy=agent)
    result = sim.run(verbose=False)
    
    print(f"Result: {result.total_trades} trades, Final P&L: {result.final_pnl:+.2f}")
    print()


def example_llm_agent():
    """Example: Run simulation with LLM agent."""
    print("Running simulation with LLM-based agent...")
    
    config = SimulationConfig(num_bonds=20, num_steps=30, random_seed=42)
    agent = LLMQuotingAgent()
    
    sim = Simulator(config=config, custom_quoting_strategy=agent)
    result = sim.run(verbose=False)
    
    print(f"Result: {result.total_trades} trades, Final P&L: {result.final_pnl:+.2f}")
    print()


def example_gymnasium_env():
    """Example: Using the Gymnasium-compatible RL environment."""
    print("Running Gymnasium-compatible RL environment...")
    
    from illiquid_market_sim import TradingEnv, EnvConfig, RewardType
    from illiquid_market_sim.baselines import AdaptiveAgent
    from illiquid_market_sim.rl import evaluate_policy
    
    # Create environment
    config = EnvConfig(
        max_episode_steps=50,
        reward_type=RewardType.RISK_ADJUSTED_PNL,
    )
    env = TradingEnv(config)
    
    # Run with adaptive baseline agent
    agent = AdaptiveAgent()
    result = evaluate_policy(env, agent.get_policy(), n_episodes=5, seed=42)
    
    print(f"Adaptive agent: mean_reward={result.mean_reward:.2f}, mean_pnl={result.mean_pnl:.2f}")
    print()


def example_benchmark_baselines():
    """Example: Benchmarking baseline agents."""
    print("Benchmarking baseline agents...")
    
    from illiquid_market_sim import TradingEnv, EnvConfig
    from illiquid_market_sim.baselines import (
        RandomAgent,
        FixedSpreadAgent,
        InventoryAwareAgent,
        AdaptiveAgent,
        ConservativeAgent,
        AggressiveAgent,
    )
    from illiquid_market_sim.rl import evaluate_policy
    
    env = TradingEnv(EnvConfig(max_episode_steps=50))
    
    agents = {
        "random": RandomAgent(),
        "fixed_50": FixedSpreadAgent(0.5),
        "inventory_aware": InventoryAwareAgent(),
        "adaptive": AdaptiveAgent(),
        "conservative": ConservativeAgent(),
        "aggressive": AggressiveAgent(),
    }
    
    print(f"{'Agent':<20} {'Mean Reward':>12} {'Mean PnL':>10}")
    print("-" * 45)
    
    for name, agent in agents.items():
        result = evaluate_policy(env, agent.get_policy(), n_episodes=5, seed=42)
        print(f"{name:<20} {result.mean_reward:>12.2f} {result.mean_pnl:>10.2f}")
    print()


def example_collect_offline_data():
    """Example: Collecting data for offline RL."""
    print("Collecting offline RL data...")
    
    from illiquid_market_sim import TradingEnv, EnvConfig
    from illiquid_market_sim.baselines import AdaptiveAgent
    from illiquid_market_sim.rl import DataCollector
    
    env = TradingEnv(EnvConfig(max_episode_steps=50))
    agent = AdaptiveAgent()
    
    collector = DataCollector(env, max_transitions=500)
    rewards = collector.collect_episodes(
        n_episodes=10,
        policy=agent.get_policy(),
        seed=42,
        verbose=False,
    )
    
    print(f"Collected {collector.n_transitions} transitions from {collector.n_episodes} episodes")
    print(f"Mean episode reward: {sum(rewards)/len(rewards):.2f}")
    
    # Save dataset
    # collector.save("data/adaptive_agent.npz")
    print()


if __name__ == '__main__':
    print("=" * 70)
    print("AI AGENT EXAMPLES FOR ILLIQUID BOND MARKET SIMULATOR")
    print("=" * 70)
    print()
    
    # Legacy examples (using custom strategies with Simulator)
    print("=== Legacy Examples (Custom Strategies) ===")
    example_ml_strategy()
    example_rl_agent()
    example_llm_agent()
    
    # New RL environment examples
    print("=== RL Environment Examples ===")
    example_gymnasium_env()
    example_benchmark_baselines()
    example_collect_offline_data()
    
    print("=" * 70)
    print("All examples completed!")
    print()
    print("Next steps:")
    print("1. See RL_GUIDE.md for comprehensive RL documentation")
    print("2. Train with Stable-Baselines3 or your preferred library")
    print("3. Collect offline data and train with offline RL")
    print("4. Run benchmarks to compare against baselines")
    print("=" * 70)
