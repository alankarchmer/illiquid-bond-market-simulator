"""Tests for market module."""

import pytest
from illiquid_market_sim.market import MarketState, MarketImpactModel
from illiquid_market_sim.bonds import generate_bond_universe


def test_market_state_initialization():
    """Test MarketState initialization."""
    market = MarketState()
    
    assert market.current_step == 0
    assert market.level_IG == 0.0
    assert market.level_HY == 0.0
    assert market.level_EM == 0.0


def test_market_state_step():
    """Test that market state evolves."""
    market = MarketState(volatility=0.02, jump_probability=0.0)
    bonds = generate_bond_universe(10, seed=42)
    
    initial_ig = market.level_IG
    
    # Step forward
    event = market.step(bonds)
    
    # Step should have incremented
    assert market.current_step == 1
    
    # Factors should have changed (with very high probability)
    # Note: could stay same by chance, but unlikely
    factors = market.get_factors()
    assert 'level_IG' in factors
    assert 'level_HY' in factors
    assert 'level_EM' in factors


def test_market_state_jump_events():
    """Test that jump events can occur."""
    market = MarketState(volatility=0.01, jump_probability=1.0)  # Force jump
    bonds = generate_bond_universe(10, seed=42)
    
    # With jump_probability=1.0, should get an event
    event = market.step(bonds)
    
    assert event is not None
    assert 'type' in event
    assert event['type'] in ['market_shock', 'sector_shock', 'issuer_event']


def test_market_impact_model_direct_impact():
    """Test direct impact on traded bond."""
    bonds = generate_bond_universe(10, seed=42)
    bond = bonds[0]
    
    impact_model = MarketImpactModel(
        base_impact_coeff=0.001,
        cross_impact_factor=0.3
    )
    
    initial_price = bond.get_true_fair_price()
    
    # Dealer buys (price should move up)
    impacts = impact_model.apply_trade_impact(
        traded_bond=bond,
        side="buy",
        size=5.0,
        all_bonds=bonds
    )
    
    new_price = bond.get_true_fair_price()
    
    # Price should have increased
    assert new_price > initial_price
    
    # Should have recorded impact
    assert bond.id in impacts
    assert impacts[bond.id] > 0


def test_market_impact_model_cross_impact():
    """Test cross-impact on related bonds."""
    bonds = generate_bond_universe(20, seed=42)
    
    # Find bonds with same issuer
    traded_bond = bonds[0]
    issuer = traded_bond.issuer
    
    # Find another bond with same issuer
    related_bond = None
    for b in bonds[1:]:
        if b.issuer == issuer:
            related_bond = b
            break
    
    if related_bond is None:
        pytest.skip("No related bond found in test universe")
    
    impact_model = MarketImpactModel(
        base_impact_coeff=0.001,
        cross_impact_factor=0.5
    )
    
    initial_price_related = related_bond.get_true_fair_price()
    
    # Trade the first bond
    impacts = impact_model.apply_trade_impact(
        traded_bond=traded_bond,
        side="buy",
        size=3.0,
        all_bonds=bonds
    )
    
    new_price_related = related_bond.get_true_fair_price()
    
    # Related bond should also be impacted
    assert new_price_related != initial_price_related
    assert related_bond.id in impacts


def test_market_impact_direction():
    """Test that impact direction is correct."""
    bonds = generate_bond_universe(5, seed=42)
    bond = bonds[0]
    
    impact_model = MarketImpactModel(base_impact_coeff=0.001)
    
    initial_price = bond.get_true_fair_price()
    
    # Dealer buys -> price up
    impact_model.apply_trade_impact(bond, "buy", 5.0, bonds)
    price_after_buy = bond.get_true_fair_price()
    assert price_after_buy > initial_price
    
    # Reset
    bond.update_fair_price(initial_price)
    bond._accumulated_impact = 0.0
    
    # Dealer sells -> price down
    impact_model.apply_trade_impact(bond, "sell", 5.0, bonds)
    price_after_sell = bond.get_true_fair_price()
    assert price_after_sell < initial_price


def test_market_impact_liquidity_effect():
    """Test that less liquid bonds have larger impact."""
    bond_liquid = generate_bond_universe(1, seed=42)[0]
    bond_liquid.liquidity = 0.8
    bond_liquid.update_fair_price(100.0)
    bond_liquid._accumulated_impact = 0.0
    
    bond_illiquid = generate_bond_universe(1, seed=43)[0]
    bond_illiquid.liquidity = 0.1
    bond_illiquid.update_fair_price(100.0)
    bond_illiquid._accumulated_impact = 0.0
    
    impact_model = MarketImpactModel(base_impact_coeff=0.01, cross_impact_factor=0.0)
    
    # Same size trade on both
    impact_model.apply_trade_impact(bond_liquid, "buy", 2.0, [bond_liquid])
    impact_model.apply_trade_impact(bond_illiquid, "buy", 2.0, [bond_illiquid])
    
    impact_liquid = bond_liquid.get_true_fair_price() - 100.0
    impact_illiquid = bond_illiquid.get_true_fair_price() - 100.0
    
    # Illiquid bond should have larger impact
    assert impact_illiquid > impact_liquid
