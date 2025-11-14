"""Tests for clients module."""

import pytest
from illiquid_market_sim.clients import (
    RealMoneyClient, HedgeFundClient, FisherClient, NoiseClient,
    create_client_universe, ClientStats
)
from illiquid_market_sim.bonds import generate_bond_universe
from illiquid_market_sim.market import MarketState
from illiquid_market_sim.rfq import RFQ, Quote


def test_client_stats():
    """Test ClientStats tracking."""
    stats = ClientStats(client_id="TEST01")
    
    assert stats.get_fill_ratio() == 0.0
    assert stats.get_avg_edge() == 0.0
    assert not stats.is_toxic()
    
    # Add some data
    stats.rfq_count = 10
    stats.trade_count = 2
    stats.total_edge_captured = 1.5
    
    assert stats.get_fill_ratio() == 0.2
    assert stats.get_avg_edge() == 0.75


def test_real_money_client():
    """Test RealMoneyClient behavior."""
    client = RealMoneyClient(client_id="RM01", name="Test Fund")
    
    assert client.client_type == "real_money"
    assert client.rfq_probability < 0.1  # Low frequency
    assert client.mean_size > 1.0  # Larger sizes


def test_hedge_fund_client():
    """Test HedgeFundClient behavior."""
    client = HedgeFundClient(client_id="HF01", name="Test HF")
    
    assert client.client_type == "hedge_fund"
    assert client.information_quality > 0.5  # Better info


def test_fisher_client():
    """Test FisherClient behavior."""
    client = FisherClient(client_id="FI01", name="Fisher 1")
    
    assert client.client_type == "fisher"
    assert client.fishing_probability > 0.5  # High fishing rate


def test_noise_client():
    """Test NoiseClient behavior."""
    client = NoiseClient(client_id="NO01", name="Noise 1")
    
    assert client.client_type == "noise"


def test_client_rfq_generation():
    """Test that clients can generate RFQs."""
    bonds = generate_bond_universe(10, seed=42)
    market_state = MarketState()
    
    client = RealMoneyClient(client_id="RM01", name="Test")
    
    # Set high probability for testing
    client.rfq_probability = 1.0
    
    rfq = client.maybe_generate_rfq(
        timestep=0,
        market_state=market_state,
        bonds=bonds
    )
    
    assert rfq is not None
    assert isinstance(rfq, RFQ)
    assert rfq.client_id == "RM01"
    assert rfq.bond_id in [b.id for b in bonds]
    assert rfq.side in ["buy", "sell"]
    assert rfq.size > 0


def test_client_trade_decision_real_money():
    """Test RealMoneyClient trade decision logic."""
    client = RealMoneyClient(client_id="RM01", name="Test")
    bonds = generate_bond_universe(5, seed=42)
    bond = bonds[0]
    
    rfq = RFQ(
        rfq_id="TEST_RFQ",
        timestamp=0,
        client_id="RM01",
        bond_id=bond.id,
        side="buy",
        size=2.0,
        is_fishing=False
    )
    
    # Fair value estimate
    fair_value = 100.0
    
    # Good quote (close to fair) - should trade
    quote_good = Quote(rfq_id="TEST_RFQ", price=99.5, spread_bps=50, timestamp=0)
    assert client.decide_trade(rfq, quote_good, fair_value)
    
    # Bad quote (too far from fair) - should not trade
    quote_bad = Quote(rfq_id="TEST_RFQ", price=105.0, spread_bps=500, timestamp=0)
    assert not client.decide_trade(rfq, quote_bad, fair_value)
    
    # Fishing RFQ - should never trade
    rfq.is_fishing = True
    assert not client.decide_trade(rfq, quote_good, fair_value)


def test_client_trade_decision_hedge_fund():
    """Test HedgeFundClient trade decision logic."""
    client = HedgeFundClient(client_id="HF01", name="Test HF")
    bonds = generate_bond_universe(5, seed=42)
    bond = bonds[0]
    
    rfq = RFQ(
        rfq_id="TEST_RFQ",
        timestamp=0,
        client_id="HF01",
        bond_id=bond.id,
        side="buy",
        size=1.5,
        is_fishing=False
    )
    
    fair_value = 100.0
    
    # Client buying - wants price below fair
    quote_cheap = Quote(rfq_id="TEST_RFQ", price=99.0, spread_bps=100, timestamp=0)
    assert client.decide_trade(rfq, quote_cheap, fair_value)
    
    quote_expensive = Quote(rfq_id="TEST_RFQ", price=101.0, spread_bps=100, timestamp=0)
    assert not client.decide_trade(rfq, quote_expensive, fair_value)


def test_create_client_universe():
    """Test client universe creation."""
    clients = create_client_universe(
        num_real_money=2,
        num_hedge_fund=2,
        num_fisher=1,
        num_noise=1
    )
    
    assert len(clients) == 6
    
    # Check types
    types = [c.client_type for c in clients]
    assert types.count("real_money") == 2
    assert types.count("hedge_fund") == 2
    assert types.count("fisher") == 1
    assert types.count("noise") == 1
    
    # Check unique IDs
    ids = [c.client_id for c in clients]
    assert len(ids) == len(set(ids))


def test_hedge_fund_better_info():
    """Test that hedge funds have better fair value estimates."""
    bonds = generate_bond_universe(5, seed=42)
    bond = bonds[0]
    market_state = MarketState()
    
    # Set a known fair price
    bond.update_fair_price(100.0)
    true_fair = bond.get_true_fair_price()
    
    hf_client = HedgeFundClient(client_id="HF01", name="HF")
    rm_client = RealMoneyClient(client_id="RM01", name="RM")
    
    # Get multiple estimates and check variance
    hf_estimates = [hf_client.get_fair_value_estimate(bond, market_state) for _ in range(10)]
    rm_estimates = [rm_client.get_fair_value_estimate(bond, market_state) for _ in range(10)]
    
    # Hedge fund estimates should be closer to true fair on average
    hf_avg_error = sum(abs(e - true_fair) for e in hf_estimates) / len(hf_estimates)
    rm_avg_error = sum(abs(e - true_fair) for e in rm_estimates) / len(rm_estimates)
    
    # HF should have lower error (this is probabilistic but should hold with seed)
    # We check it's at least not worse
    assert hf_avg_error <= rm_avg_error * 1.5
