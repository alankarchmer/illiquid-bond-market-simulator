"""Tests for bonds module."""

import pytest
from illiquid_market_sim.bonds import Bond, generate_bond_universe, get_bonds_by_sector


def test_bond_creation():
    """Test basic bond creation."""
    bond = Bond(
        id="TEST001",
        issuer="Test Corp",
        sector="HY",
        rating="BB",
        maturity_years=5.0,
        liquidity=0.5,
        volatility=0.02,
        base_spread=300.0
    )
    
    assert bond.id == "TEST001"
    assert bond.sector == "HY"
    assert bond.get_true_fair_price() == 100.0  # Default


def test_bond_fair_price_updates():
    """Test that bond fair prices can be updated."""
    bond = Bond(
        id="TEST001",
        issuer="Test Corp",
        sector="IG",
        rating="BBB",
        maturity_years=5.0,
        liquidity=0.7,
        volatility=0.01,
        base_spread=150.0
    )
    
    initial_price = bond.get_true_fair_price()
    assert initial_price == 100.0
    
    # Update fair price
    bond.update_fair_price(102.5)
    assert bond.get_true_fair_price() == 102.5


def test_bond_impact():
    """Test that market impact is applied correctly."""
    bond = Bond(
        id="TEST001",
        issuer="Test Corp",
        sector="HY",
        rating="B",
        maturity_years=3.0,
        liquidity=0.3,
        volatility=0.015,
        base_spread=500.0
    )
    
    initial_price = bond.get_true_fair_price()
    
    # Apply impact
    bond.apply_impact(1.5)
    new_price = bond.get_true_fair_price()
    
    assert new_price == initial_price + 1.5


def test_bond_trade_recording():
    """Test that trades are recorded."""
    bond = Bond(
        id="TEST001",
        issuer="Test Corp",
        sector="IG",
        rating="A",
        maturity_years=10.0,
        liquidity=0.8,
        volatility=0.008,
        base_spread=100.0
    )
    
    assert bond.get_last_traded_price() is None
    
    bond.record_trade(101.5)
    assert bond.get_last_traded_price() == 101.5


def test_generate_bond_universe():
    """Test bond universe generation."""
    n_bonds = 20
    bonds = generate_bond_universe(n_bonds, seed=42)
    
    assert len(bonds) == n_bonds
    assert all(isinstance(b, Bond) for b in bonds)
    
    # Check diversity
    sectors = set(b.sector for b in bonds)
    assert len(sectors) > 1  # Should have multiple sectors
    
    # Check IDs are unique
    ids = [b.id for b in bonds]
    assert len(ids) == len(set(ids))


def test_bond_universe_reproducibility():
    """Test that same seed produces same universe."""
    bonds1 = generate_bond_universe(10, seed=123)
    bonds2 = generate_bond_universe(10, seed=123)
    
    for b1, b2 in zip(bonds1, bonds2):
        assert b1.id == b2.id
        assert b1.sector == b2.sector
        assert b1.rating == b2.rating


def test_get_bonds_by_sector():
    """Test filtering bonds by sector."""
    bonds = generate_bond_universe(30, seed=42)
    
    hy_bonds = get_bonds_by_sector(bonds, "HY")
    assert all(b.sector == "HY" for b in hy_bonds)
    assert len(hy_bonds) > 0


def test_bond_naive_mid():
    """Test naive mid calculation."""
    bond = Bond(
        id="TEST001",
        issuer="Test Corp",
        sector="IG",
        rating="AA",
        maturity_years=7.0,
        liquidity=0.75,
        volatility=0.005,
        base_spread=75.0
    )
    
    market_factors = {"level_IG": 0.5, "level_HY": 1.0, "level_EM": 1.5}
    
    # Should return a price around 100 + sector adjustment + noise
    mid = bond.get_naive_mid(market_factors)
    assert 99.0 < mid < 102.0  # Reasonable range with noise
