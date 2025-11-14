#!/usr/bin/env python3
"""
Command-line interface for the illiquid bond market simulator.

Provides two modes:
1. Simulation mode: Run automated simulations with rule-based dealer
2. Game mode: Human plays as the dealer, making quoting decisions
"""

import argparse
import sys
from typing import Optional

from illiquid_market_sim.config import SimulationConfig
from illiquid_market_sim.simulation import Simulator
from illiquid_market_sim.agent import ManualQuotingStrategy
from illiquid_market_sim.metrics import summarize_simulation


def run_simulation_mode(
    num_steps: int,
    num_bonds: int,
    verbose: bool,
    random_seed: Optional[int]
) -> None:
    """
    Run automated simulation with rule-based dealer.
    
    Args:
        num_steps: Number of simulation steps
        num_bonds: Number of bonds in universe
        verbose: Whether to print detailed progress
        random_seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("ILLIQUID BOND MARKET SIMULATOR - SIMULATION MODE")
    print("=" * 70)
    print()
    
    # Create config
    config = SimulationConfig(
        num_bonds=num_bonds,
        num_steps=num_steps,
        random_seed=random_seed if random_seed is not None else 42
    )
    
    print("Configuration:")
    print(f"  Bonds: {config.num_bonds}")
    print(f"  Steps: {config.num_steps}")
    print(f"  Clients: {config.num_real_money_clients} real money, "
          f"{config.num_hedge_fund_clients} hedge funds, "
          f"{config.num_fisher_clients} fishers, "
          f"{config.num_noise_clients} noise")
    print(f"  Seed: {config.random_seed}")
    print()
    
    # Run simulation
    simulator = Simulator(config=config)
    result = simulator.run(verbose=verbose)
    
    # Print summary (already printed if verbose=True)
    if not verbose:
        print(summarize_simulation(result))
    
    # Additional analytics
    print("\nADDITIONAL ANALYTICS")
    print("-" * 70)
    
    if result.pnl_history:
        from illiquid_market_sim.metrics import calculate_sharpe_ratio, calculate_max_drawdown
        
        sharpe = calculate_sharpe_ratio(result.pnl_history)
        max_dd = calculate_max_drawdown(result.pnl_history)
        
        print(f"Sharpe Ratio:      {sharpe:.2f}")
        print(f"Max Drawdown:      {max_dd:.2f}")
    
    print()
    print("Simulation complete!")


def run_game_mode(
    num_steps: int,
    num_bonds: int,
    random_seed: Optional[int]
) -> None:
    """
    Run interactive game mode where human plays as dealer.
    
    Args:
        num_steps: Number of simulation steps
        num_bonds: Number of bonds in universe
        random_seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("ILLIQUID BOND MARKET SIMULATOR - GAME MODE")
    print("=" * 70)
    print()
    print("Welcome! You are the dealer.")
    print("For each RFQ, you'll decide what price to quote.")
    print("Your goal: maximize P&L while managing risk.")
    print()
    print("Commands:")
    print("  - Enter a price to quote that price")
    print("  - Type 'pass' or 'p' to pass on the RFQ (quote very wide)")
    print()
    input("Press Enter to start...")
    print()
    
    # Create config
    config = SimulationConfig(
        num_bonds=num_bonds,
        num_steps=num_steps,
        random_seed=random_seed if random_seed is not None else 42,
        # Lower RFQ probability for game mode to avoid overwhelming player
        rfq_prob_per_client=0.08
    )
    
    # Create simulator with manual quoting strategy
    manual_strategy = ManualQuotingStrategy()
    simulator = Simulator(config=config, custom_quoting_strategy=manual_strategy)
    
    # Run simulation (non-verbose since we're interactive)
    print(f"Starting game: {num_steps} steps, {num_bonds} bonds")
    print("=" * 70)
    print()
    
    result = simulator.run(num_steps=num_steps, verbose=False)
    
    # Show final results
    print("\n" + "=" * 70)
    print("GAME OVER!")
    print("=" * 70)
    print()
    print(summarize_simulation(result))
    
    # Performance evaluation
    print("\nYOUR PERFORMANCE")
    print("-" * 70)
    
    if result.final_pnl > 0:
        print(f"✓ Profitable! Final P&L: +{result.final_pnl:.2f}")
    else:
        print(f"✗ Unprofitable. Final P&L: {result.final_pnl:.2f}")
    
    if result.fill_ratio > 0.5:
        print(f"✓ Good fill ratio: {result.fill_ratio:.1%}")
    elif result.fill_ratio > 0.3:
        print(f"~ Moderate fill ratio: {result.fill_ratio:.1%}")
    else:
        print(f"✗ Low fill ratio: {result.fill_ratio:.1%} (too wide?)")
    
    if result.inventory_risk < 20:
        print(f"✓ Well-managed inventory: {result.inventory_risk:.1f}")
    else:
        print(f"⚠ High inventory risk: {result.inventory_risk:.1f}")
    
    print()
    print("Thanks for playing!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Illiquid Bond Market Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a quick simulation
  python cli.py --steps 50
  
  # Run with more bonds and verbose output
  python cli.py --steps 100 --bonds 100 --verbose
  
  # Play in game mode
  python cli.py --game --steps 30
  
  # Run with specific seed for reproducibility
  python cli.py --steps 100 --seed 123
        """
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of simulation steps (default: 100)'
    )
    
    parser.add_argument(
        '--bonds',
        type=int,
        default=50,
        help='Number of bonds in universe (default: 50)'
    )
    
    parser.add_argument(
        '--game',
        action='store_true',
        help='Run in interactive game mode (you are the dealer)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (simulation mode only)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    try:
        if args.game:
            run_game_mode(
                num_steps=args.steps,
                num_bonds=args.bonds,
                random_seed=args.seed
            )
        else:
            run_simulation_mode(
                num_steps=args.steps,
                num_bonds=args.bonds,
                verbose=args.verbose,
                random_seed=args.seed
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
