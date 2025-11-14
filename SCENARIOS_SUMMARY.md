# Scenario System Implementation Summary

## Overview

A comprehensive scenario system has been added to the illiquid bond market simulator, providing 15 distinct market regimes that can be selected when running simulations.

## What Was Implemented

### 1. Core Scenario Module (`scenarios.py`)

Created a new module with:
- **`ScenarioConfig`** dataclass containing all scenario parameters:
  - Market dynamics (volatility, drift, shock probabilities)
  - Flow characteristics (RFQ volume, size, buy/sell skew)
  - Client mix (fractions of each client type)
  - Impact parameters (liquidity, cross-issuer, cross-sector)
  - Support for time-varying parameters (regime shifts)

- **`get_scenarios()`** function returning 15 predefined scenarios
- **`list_scenarios()`** function for formatted scenario listing

### 2. Enhanced Market Dynamics (`market.py`)

Updated `MarketState` to support:
- Persistent spread drift (positive = widening, negative = tightening)
- Separate shock probabilities for:
  - Market-wide shocks
  - Sector-specific shocks
  - Idiosyncratic issuer events

Updated `MarketImpactModel` to support:
- Liquidity multiplier (scenario-specific liquidity conditions)
- Configurable cross-issuer and cross-sector impact

### 3. Enhanced Client System (`clients.py`)

Added:
- **Buy/sell skew** parameter to bias RFQ sides
- **`create_clients_from_scenario()`** function to generate clients based on scenario specifications
- Dynamic RFQ probability and size scaling

### 4. Scenario-Aware Simulator (`simulation.py`)

Updated `Simulator` to:
- Accept `ScenarioConfig` on initialization
- Initialize all components using scenario parameters
- Support regime shift scenarios via callback mechanism
- Apply initial positions for scenarios requiring pre-existing inventory

### 5. CLI Integration (`cli.py`)

Added:
- `--list-scenarios` flag to display all available scenarios
- `--scenario NAME` flag to run a specific scenario
- Default to 'quiet_market' if no scenario specified
- Enhanced output showing scenario parameters and description

## The 15 Scenarios

### Market Stress & Regime

1. **quiet_market** - Stable spreads, low vol, moderate flow (calibration baseline)
2. **grind_tighter** - Risk-on rally with tightening spreads, buy bias
3. **grind_wider** - Risk-off drift with widening spreads, sell pressure
4. **credit_shock** - Mini-crisis with spiking vol, panic selling, liquidity dry-up
5. **sector_blowup** - One sector stressed, others stable
6. **issuer_event** - Idiosyncratic issuer-specific events (downgrades, rumors)
7. **regime_shift** - Quiet first half → stressed second half

### Flow & Client Mix

8. **fisher_onslaught** - High volume of fishing RFQs, low fill ratio
9. **informed_flow** - Toxic hedge fund flow anticipating moves
10. **big_real_money** - Large, lumpy, price-insensitive trades
11. **etf_rebalance** - Many small basket RFQs, low information content

### Liquidity Conditions

12. **liquidity_dryup** - Thin markets, high impact per trade
13. **month_end** - Distorted flow around benchmarks

### Position-Specific

14. **short_squeeze** - Start short a sector, tightening spreads force covering
15. **inventory_overhang** - Long illiquid bonds in risk-off environment

## Usage Examples

```bash
# List all scenarios
python3 cli.py --list-scenarios

# Run with default scenario (quiet_market)
python3 cli.py --steps 100

# Run a specific scenario
python3 cli.py --scenario credit_shock --steps 200 --verbose

# Compare scenarios
python3 cli.py --scenario grind_tighter --steps 100 --seed 42
python3 cli.py --scenario grind_wider --steps 100 --seed 42

# Test regime shift
python3 cli.py --scenario regime_shift --steps 100 --verbose
```

## Key Features

### Realistic Market Dynamics
- Scenarios model real-world regimes (risk-on/risk-off, crises, rebalancing)
- Persistent drift captures trend behavior
- Multiple shock types (market, sector, issuer) create realistic volatility clustering

### Flow Realism
- Buy/sell skew captures directional pressure
- Client mix varies by scenario (e.g., more hedge funds in "informed_flow")
- RFQ volume and size scale appropriately

### Testable Edge Cases
- Extreme stress scenarios for robustness testing
- Fisher-dominated scenarios for adverse selection
- Liquidity dry-ups for impact modeling

### Extensibility
- Easy to add new scenarios by creating `ScenarioConfig` instances
- Regime shift callback mechanism supports time-varying parameters
- Initial position support for testing inventory management

## Testing

All existing tests pass with the new implementation:
- 34 tests passed, 1 skipped
- Backward compatibility maintained (scenarios are optional)
- New scenario system fully integrated without breaking existing functionality

## Code Quality

- Type hints throughout
- Comprehensive docstrings
- Clean separation of concerns (scenarios, market, clients, simulation)
- Dataclasses for clean configuration
- No breaking changes to existing API
