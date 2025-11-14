# Quick Start Guide

Get up and running with the Illiquid Bond Market Simulator in 5 minutes!

## Installation

```bash
# Ensure you have Python 3.11+
python3 --version

# Install optional dependencies (pandas, matplotlib)
pip install -r requirements.txt

# Or skip dependencies - the simulator works with standard library only
```

## Run Your First Simulation

```bash
# Basic simulation (100 steps, 50 bonds)
python3 cli.py

# Quick test (20 steps, 10 bonds)
python3 cli.py --steps 20 --bonds 10

# Verbose mode to see what's happening
python3 cli.py --steps 30 --verbose
```

## Play the Game

```bash
# Interactive mode - YOU are the dealer!
python3 cli.py --game --steps 20

# In game mode:
# - You'll see RFQs one by one
# - Enter your quote price for each
# - Type 'pass' to skip an RFQ
# - Try to maximize P&L while managing risk
```

## Programmatic Usage

Create a file `my_simulation.py`:

```python
from illiquid_market_sim.simulation import Simulator
from illiquid_market_sim.config import SimulationConfig
from illiquid_market_sim.metrics import summarize_simulation

# Configure
config = SimulationConfig(
    num_bonds=30,
    num_steps=50,
    random_seed=42
)

# Run
sim = Simulator(config=config)
result = sim.run(verbose=False)

# Analyze
print(summarize_simulation(result))
print(f"\nFinal P&L: {result.final_pnl:.2f}")
print(f"Fill Ratio: {result.fill_ratio:.1%}")
```

Run it:
```bash
python3 my_simulation.py
```

## Customize Client Mix

```python
config = SimulationConfig(
    num_bonds=50,
    num_steps=100,
    # More aggressive client mix
    num_real_money_clients=1,
    num_hedge_fund_clients=5,  # More informed traders
    num_fisher_clients=3,       # More price discovery
    num_noise_clients=1
)
```

## Adjust Dealer Strategy

```python
from illiquid_market_sim.agent import RuleBasedQuotingStrategy

# Create custom strategy with wider spreads
custom_strategy = RuleBasedQuotingStrategy(
    base_spread_bps=100.0,  # Wider base spread
    inventory_risk_penalty=0.02,  # More risk averse
    toxic_client_spread_multiplier=2.0  # Widen more for toxic clients
)

# Use it
sim = Simulator(config=config, custom_quoting_strategy=custom_strategy)
result = sim.run()
```

## Run Tests

```bash
# Install pytest
pip install pytest

# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_simulation.py -v

# Run with coverage
pip install pytest-cov
python3 -m pytest tests/ --cov=illiquid_market_sim
```

## Understanding the Output

```
P&L BREAKDOWN
----------------------------------------------------------------------
Final Total P&L:   +2.66
  Realized:        +0.97     <- Closed positions
  Unrealized:      +1.69     <- Open positions (MTM)
Impact Cost:       0.70      <- Cost of moving the market
Inventory Risk:    9.83      <- Sum of absolute positions
```

```
CLIENT BREAKDOWN
----------------------------------------------------------------------
Client          Type         RFQs  Trades  Fill%   Avg Edge
----------------------------------------------------------------------
HF01            hedge_fund   12    4       33%     +0.85    <- Informed!
FI01            fisher       15    1       7%      +2.10    <- Just fishing
RM01            real_money   5     4       80%     -0.25    <- Good client
```

**Key Insights:**
- **Hedge Funds**: Low fill ratio + high edge = they're picking you off
- **Fishers**: Very low fill ratio = widen spreads for them
- **Real Money**: High fill ratio + negative edge = good profitable flow

## Tips for Game Mode

1. **Check liquidity**: Less liquid bonds need wider spreads
2. **Watch inventory**: If you're already long, quote lower bids; if short, higher offers
3. **Track clients**: Widen spreads for fishers and informed traders
4. **Balance P&L and fills**: Too wide = no trades, too tight = losses

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the codebase in `illiquid_market_sim/`
- Modify client behaviors in `clients.py`
- Build custom quoting strategies
- Hook in your AI agents!

## Common Issues

**"python: command not found"**
→ Use `python3` instead of `python`

**"No module named pytest"**
→ Install with `pip install pytest`

**Tests fail**
→ Ensure you're using Python 3.11+ and have latest code

**Simulation seems random**
→ Use `--seed 42` for reproducible results

---

Have fun trading! 📊📈
