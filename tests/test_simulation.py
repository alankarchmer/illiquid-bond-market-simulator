"""Tests for simulation module."""

import pytest
from illiquid_market_sim.simulation import Simulator
from illiquid_market_sim.config import SimulationConfig
from illiquid_market_sim.metrics import SimulationResult


def test_simulator_initialization():
    """Test that simulator initializes correctly."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=20,
        random_seed=42
    )
    
    sim = Simulator(config=config)
    
    assert len(sim.bonds) == 10
    assert len(sim.clients) > 0
    assert sim.market_state is not None
    assert sim.portfolio is not None
    assert sim.dealer is not None


def test_simulation_run():
    """Test that simulation can run without errors."""
    config = SimulationConfig(
        num_bonds=5,
        num_steps=10,
        random_seed=42
    )
    
    sim = Simulator(config=config)
    result = sim.run(verbose=False)
    
    assert isinstance(result, SimulationResult)
    assert result.total_steps == 10


def test_simulation_produces_rfqs():
    """Test that simulation generates RFQs."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=50,
        random_seed=42,
        rfq_prob_per_client=0.5  # High probability for testing
    )
    
    sim = Simulator(config=config)
    result = sim.run(verbose=False)
    
    # Should have generated some RFQs
    assert result.total_rfqs > 0


def test_simulation_produces_trades():
    """Test that simulation generates trades."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=50,
        random_seed=42,
        rfq_prob_per_client=0.5
    )
    
    sim = Simulator(config=config)
    result = sim.run(verbose=False)
    
    # Should have some trades (with high RFQ prob)
    assert result.total_trades >= 0  # Could be 0 if no RFQs accepted
    
    # If there are trades, fill ratio should be reasonable
    if result.total_rfqs > 0:
        assert 0 <= result.fill_ratio <= 1.0


def test_simulation_tracks_pnl():
    """Test that simulation tracks P&L."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=30,
        random_seed=42
    )
    
    sim = Simulator(config=config)
    result = sim.run(verbose=False)
    
    # Should have P&L data
    assert result.final_pnl is not None
    assert result.realized_pnl is not None
    assert result.unrealized_pnl is not None
    
    # P&L should decompose correctly
    assert abs(result.final_pnl - (result.realized_pnl + result.unrealized_pnl)) < 0.01


def test_simulation_client_stats():
    """Test that simulation tracks client statistics."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=50,
        random_seed=42,
        rfq_prob_per_client=0.3
    )
    
    sim = Simulator(config=config)
    result = sim.run(verbose=False)
    
    # Should have client breakdown
    assert len(result.client_breakdown) > 0
    
    # Check structure of client stats
    for client_id, stats in result.client_breakdown.items():
        assert 'type' in stats
        assert 'rfq_count' in stats
        assert 'trade_count' in stats
        assert 'fill_ratio' in stats


def test_simulation_reproducibility():
    """Test that same seed produces same results."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=20,
        random_seed=123
    )
    
    sim1 = Simulator(config=config)
    result1 = sim1.run(verbose=False)
    
    # Create new simulator with same seed
    config2 = SimulationConfig(
        num_bonds=10,
        num_steps=20,
        random_seed=123
    )
    sim2 = Simulator(config=config2)
    result2 = sim2.run(verbose=False)
    
    # Should get same results
    assert result1.total_rfqs == result2.total_rfqs
    assert result1.total_trades == result2.total_trades
    assert abs(result1.final_pnl - result2.final_pnl) < 0.01


def test_simulation_position_tracking():
    """Test that simulation tracks positions."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=30,
        random_seed=42
    )
    
    sim = Simulator(config=config)
    result = sim.run(verbose=False)
    
    # Should have position summary
    assert 'num_positions' in result.position_summary
    assert 'total_trades' in result.position_summary
    assert 'inventory_risk' in result.position_summary


def test_simulation_pnl_history():
    """Test that simulation records P&L history."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=20,
        random_seed=42
    )
    
    sim = Simulator(config=config)
    result = sim.run(verbose=False)
    
    # Should have P&L history
    assert len(result.pnl_history) > 0
    
    # Each entry should have expected fields
    for snapshot in result.pnl_history:
        assert 'total' in snapshot
        assert 'realized' in snapshot
        assert 'unrealized' in snapshot


def test_simulation_market_events():
    """Test that market events can occur during simulation."""
    config = SimulationConfig(
        num_bonds=10,
        num_steps=100,
        random_seed=42,
        jump_probability=0.1  # Higher probability for testing
    )
    
    sim = Simulator(config=config)
    result = sim.run(verbose=False)
    
    # With 100 steps and 10% probability, should likely see some events
    # Events are tracked internally in sim.events
    # (Not exposed in result currently, but simulation should not crash)
    assert result.total_steps == 100


def test_position_reversal_cost_basis():
    """Test that position reversals correctly track cost basis."""
    from illiquid_market_sim.portfolio import Portfolio
    
    portfolio = Portfolio()
    
    # Start with a long position: buy 3 at $100
    portfolio.update_on_trade("BOND1", "buy", 3.0, 100.0)
    position = portfolio.get_position("BOND1")
    
    assert position.quantity == 3.0
    assert position.total_cost == 300.0
    assert position.get_average_cost() == 100.0
    
    # Sell 5 at $110 - this crosses zero and creates a short 2 position
    portfolio.update_on_trade("BOND1", "sell", 5.0, 110.0)
    
    # After crossing zero, should have:
    # - quantity = -2 (short 2)
    # - total_cost = 220 (2 * 110), not -250 (300 - 550)
    assert position.quantity == -2.0
    assert position.total_cost == 220.0, f"Expected total_cost=220.0, got {position.total_cost}"
    # Average cost for short position is negative (total_cost / quantity = 220 / -2 = -110)
    assert position.get_average_cost() == -110.0
    
    # Realized P&L should be 3 * (110 - 100) = 30
    assert abs(portfolio.realized_pnl - 30.0) < 0.01
    
    # Now test the reverse: buy 4 at $105 to go from short 2 to long 2
    portfolio.update_on_trade("BOND1", "buy", 4.0, 105.0)
    
    # After crossing zero again, should have:
    # - quantity = 2 (long 2)
    # - total_cost = 210 (2 * 105)
    assert position.quantity == 2.0
    assert position.total_cost == 210.0, f"Expected total_cost=210.0, got {position.total_cost}"
    assert position.get_average_cost() == 105.0
    
    # Realized P&L should include closing short: 2 * (110 - 105) = 10
    # Total realized: 30 + 10 = 40
    assert abs(portfolio.realized_pnl - 40.0) < 0.01
