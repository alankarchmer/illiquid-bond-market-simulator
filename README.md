# Illiquid Bond Market Simulator

A self-contained Python simulation of an illiquid corporate bond market with multiple synthetic client types, dealer quoting logic, market impact modeling, and P&L tracking. Designed for both automated simulation and human-playable trading games, with the ability to plug in AI agents for dealer decision-making.

## Overview

This simulator models a realistic illiquid bond market where:

- **Bonds** have hidden "true fair values" that evolve with market factors
- **Clients** (real money, hedge funds, fishers, noise traders) send RFQs with different behavioral patterns
- **Dealer** (you, or an automated agent) responds with quotes
- **Market impact** from filled trades affects prices across the portfolio
- **P&L tracking** shows realized, unrealized, and impact-driven P&L

The simulator is built with clean architecture and extensibility in mind, making it easy to:
- Run automated simulations with rule-based dealer logic
- Play interactively as the dealer (game mode)
- Later plug in AI agents (RL, LLMs, etc.) to make quoting decisions

## Features

### Bond Universe
- Synthetic bonds with sectors (IG, HY, EM), ratings, maturities, and liquidity levels
- Hidden fair values that evolve with market factors and credit events
- Realistic volatility and spread curves

### Client Types
- **Real Money**: Low RFQ frequency, large sizes, rarely fishes
- **Hedge Funds**: More frequent RFQs, better information, trades aggressively on edge
- **Fishers**: High RFQ frequency, mostly price discovery, rarely trades
- **Noise Traders**: Random behavior

### Market Dynamics
- Factor-driven price evolution (sector-level factors)
- Jump events: sector shocks and issuer downgrades
- Market impact model: trades move prices of the bond and related bonds (same issuer/sector)

### Dealer Agent
- Rule-based quoting strategy (default)
- Adjusts spreads for:
  - Bond liquidity
  - Inventory risk
  - Client toxicity
- Pluggable architecture for custom strategies (including manual/human-in-the-loop)

### Metrics & Analysis
- Realized and unrealized P&L
- Market impact cost estimation
- Per-client statistics (fill ratios, edge captured)
- Sharpe ratio and max drawdown
- Full trade and RFQ history

## Installation

### Requirements
- Python 3.11 or higher
- Standard library (core functionality)
- Optional: pandas, matplotlib (for enhanced analysis)

### Setup

```bash
# Clone or download the project
cd illiquid-bond-market-simulator

# Install dependencies (minimal)
pip install -r requirements.txt

# Or if you don't need pandas/matplotlib
# No dependencies needed - just run with Python 3.11+
```

## Usage

### Quick Start: Run a Simulation

```bash
# Run a basic 100-step simulation
python cli.py

# Run with verbose output
python cli.py --verbose

# Run a shorter simulation with more bonds
python cli.py --steps 50 --bonds 100

# Run with a specific seed for reproducibility
python cli.py --steps 100 --seed 42
```

### Game Mode: Play as the Dealer

```bash
# Interactive mode - you make the quoting decisions!
python cli.py --game --steps 30

# Longer game with more bonds
python cli.py --game --steps 50 --bonds 75
```

In game mode:
1. You'll see each RFQ with bond details and your current position
2. You enter a quote price (or type 'pass' to skip)
3. The client decides whether to trade based on your quote
4. At the end, you'll see your P&L, fill ratios, and performance metrics

### Programmatic Usage

```python
from illiquid_market_sim.simulation import Simulator
from illiquid_market_sim.config import SimulationConfig
from illiquid_market_sim.metrics import summarize_simulation

# Create a custom configuration
config = SimulationConfig(
    num_bonds=50,
    num_steps=100,
    random_seed=42,
    base_spread_bps=50,
    rfq_prob_per_client=0.15
)

# Run simulation
simulator = Simulator(config=config)
result = simulator.run(verbose=True)

# Print summary
print(summarize_simulation(result))

# Access detailed results
print(f"Final P&L: {result.final_pnl:.2f}")
print(f"Fill Ratio: {result.fill_ratio:.1%}")
print(f"Total Trades: {result.total_trades}")
```

### Plug in Custom Quoting Strategy

```python
from illiquid_market_sim.simulation import Simulator
from illiquid_market_sim.agent import QuotingStrategy
from illiquid_market_sim.rfq import Quote

# Define your custom strategy
class MyCustomStrategy:
    def generate_quote(self, rfq, bond, portfolio, market_state, client_stats):
        # Your logic here
        # Could be ML model, heuristics, etc.
        price = 100.0  # Simplified example
        return Quote(
            rfq_id=rfq.rfq_id,
            price=price,
            spread_bps=50.0,
            timestamp=rfq.timestamp
        )

# Use custom strategy
simulator = Simulator(custom_quoting_strategy=MyCustomStrategy())
result = simulator.run()
```

## Project Structure

```
illiquid_market_sim/
├── __init__.py
├── bonds.py          # Bond model and universe generation
├── clients.py        # Client types and behavior
├── rfq.py            # RFQ, Quote, and Trade dataclasses
├── market.py         # Market state, factors, and impact model
├── portfolio.py      # Position and portfolio tracking
├── agent.py          # Dealer quoting strategies
├── metrics.py        # P&L tracking and analysis
├── config.py         # Configuration management
└── simulation.py     # Simulation orchestration

tests/
├── test_bonds.py
├── test_clients.py
├── test_market.py
└── test_simulation.py

cli.py                # Command-line interface
requirements.txt
README.md
```

## Key Concepts

### Hidden Fair Values

Each bond has a "true" fair value that evolves with market factors, but is hidden from the dealer. The dealer must estimate fair value using:
- Last traded prices
- Sector/rating spreads
- Market factors (observed)

This information asymmetry is key to the game - some clients (hedge funds) have better information than the dealer.

### Market Impact

When a trade executes:
1. **Direct impact**: The traded bond's price moves based on size and liquidity
2. **Cross impact**: Related bonds (same issuer, same sector) also move
3. **Portfolio effect**: Your entire book is marked at the new prices

This models the reality that large trades in illiquid bonds move the market, affecting your whole portfolio.

### Client Behavior

Clients differ in their:
- **RFQ frequency**: How often they request quotes
- **Fishing probability**: How often they're just price-checking vs. actually wanting to trade
- **Information quality**: How well they estimate fair value
- **Trade logic**: When they accept vs. reject quotes

The dealer can track client statistics (fill ratios, edge captured) and adjust spreads accordingly.

### P&L Decomposition

P&L is broken down into:
- **Realized**: From closing out positions
- **Unrealized**: Mark-to-market on open positions
- **Impact cost**: Estimated cost of moving the market

This helps understand not just total P&L, but where it comes from.

## Configuration Options

Key configuration parameters (see `config.py`):

```python
SimulationConfig(
    # Universe
    num_bonds=50,                    # Number of bonds
    num_steps=100,                   # Simulation length
    random_seed=42,                  # Reproducibility
    
    # Market dynamics
    market_volatility=0.02,          # Daily volatility
    jump_probability=0.05,           # Credit event probability
    
    # Market impact
    base_impact_coeff=0.001,         # Impact per unit size
    cross_impact_factor=0.3,         # Spillover to related bonds
    
    # Dealer
    base_spread_bps=50,              # Base spread in bps
    inventory_risk_penalty=0.01,     # Penalty per unit inventory
    
    # Clients
    num_real_money_clients=3,
    num_hedge_fund_clients=2,
    num_fisher_clients=2,
    num_noise_clients=3,
    
    # RFQ generation
    rfq_prob_per_client=0.1,         # RFQ probability per step per client
)
```

## Testing

Run tests to verify everything works:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=illiquid_market_sim

# Run specific test file
pytest tests/test_simulation.py -v
```

## Example Output

### Simulation Mode

```
======================================================================
SIMULATION SUMMARY
======================================================================

Total Steps:       100
Total RFQs:        82
Total Trades:      23
Fill Ratio:        28.0%

P&L BREAKDOWN
----------------------------------------------------------------------
Final Total P&L:   +2.34
  Realized:        +1.85
  Unrealized:      +0.49
Impact Cost:       0.12
Inventory Risk:    8.50

CLIENT BREAKDOWN
----------------------------------------------------------------------
Client          Type         RFQs     Trades   Fill%    Avg Edge  
----------------------------------------------------------------------
HF01            hedge_fund   18       7        38.9%    +0.85     
HF02            hedge_fund   15       5        33.3%    +0.62     
RM01            real_money   8        4        50.0%    -0.15     
...

POSITION SUMMARY
----------------------------------------------------------------------
Active Positions:  12
Total Trades:      23
Inventory Risk:    8.50
```

### Game Mode

```
============================================================
RFQ: RFQ[HF01_RFQ0042] HF01 wants to BUY 2.50 of BOND023 @ t=15
Bond: BOND023 (RetailCo 7, HY, B)
Maturity: 7.2y, Liquidity: 0.25
Naive mid estimate: 101.25
Current position: -1.50
Client stats: 8 RFQs, 3 trades, fill ratio: 37.5%
============================================================
Enter your quote price (or 'pass' to skip): 102.5

[TRADE] Trade[T00012] BOND023: dealer sell 2.50 @ 102.50
  Impact: +0.18 on BOND023

...

GAME OVER!
YOUR PERFORMANCE
----------------------------------------------------------------------
✓ Profitable! Final P&L: +3.72
✓ Good fill ratio: 42.3%
✓ Well-managed inventory: 12.3
```

## Reinforcement Learning

The simulator includes a complete RL environment compatible with Gymnasium/PettingZoo:

```python
from illiquid_market_sim import TradingEnv, EnvConfig

# Create environment
env = TradingEnv(EnvConfig(max_episode_steps=100))

# Standard Gymnasium API
obs, info = env.reset(seed=42)
action = env.action_space_sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Key RL Features

- **Gymnasium-compatible API**: Works with Stable-Baselines3, CleanRL, etc.
- **Multiple action types**: Continuous spread, discrete spread, accept/reject
- **Configurable rewards**: PnL, risk-adjusted PnL, execution quality, Sharpe
- **Multi-agent mode**: PettingZoo-style API for competitive market making
- **Baseline agents**: Random, fixed spread, inventory-aware, adaptive, etc.
- **Offline RL support**: Data collection and loading utilities

See **[RL_GUIDE.md](RL_GUIDE.md)** for comprehensive documentation.

### Quick RL Example

```python
from illiquid_market_sim import TradingEnv, EnvConfig, RewardType
from illiquid_market_sim.baselines import AdaptiveAgent
from illiquid_market_sim.rl import evaluate_policy

# Create environment with risk-adjusted rewards
config = EnvConfig(
    max_episode_steps=100,
    reward_type=RewardType.RISK_ADJUSTED_PNL,
)
env = TradingEnv(config)

# Evaluate a baseline agent
agent = AdaptiveAgent()
result = evaluate_policy(env, agent.get_policy(), n_episodes=10)
print(f"Mean reward: {result.mean_reward:.2f}")
```

## Future Enhancements

Potential extensions:
- **Advanced RL**: Curriculum learning, multi-objective rewards
- **LLM Integration**: Language model-based trading strategies
- **Visualization**: Real-time P&L charts, position heatmaps
- **Historical Replay**: Replay actual market scenarios

## Design Philosophy

This simulator prioritizes:
1. **Clarity**: Clean, readable code with type hints and docstrings
2. **Modularity**: Easy to swap components (client types, quoting strategies, impact models)
3. **Extensibility**: Structured for adding AI agents, new client types, complex features
4. **Realism**: Simplified but realistic models of illiquid bond market dynamics
5. **Playability**: Fun to play as a human, informative as a learning tool

## License

MIT License - feel free to use, modify, and extend!

## Contributing

Contributions welcome! Areas for improvement:
- More sophisticated client behaviors
- Alternative impact models
- Performance optimizations
- Additional metrics and analytics
- Better visualization
- Documentation and examples

## Contact

For questions, suggestions, or to share your results, please open an issue on GitHub.

---

**Happy trading!** May your spreads be wide and your fills be profitable. 📈
