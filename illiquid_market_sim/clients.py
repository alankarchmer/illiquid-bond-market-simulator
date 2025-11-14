"""
Client models with different trading behaviors.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from abc import ABC, abstractmethod

from illiquid_market_sim.bonds import Bond
from illiquid_market_sim.market import MarketState
from illiquid_market_sim.rfq import RFQ, Quote


@dataclass
class ClientStats:
    """Statistics tracked per client for dealer to use."""
    
    client_id: str
    rfq_count: int = 0
    trade_count: int = 0
    total_edge_captured: float = 0.0  # How much profit client made vs fair
    
    def get_fill_ratio(self) -> float:
        """Fraction of RFQs that resulted in trades."""
        return self.trade_count / self.rfq_count if self.rfq_count > 0 else 0.0
    
    def get_avg_edge(self) -> float:
        """Average edge per trade."""
        return self.total_edge_captured / self.trade_count if self.trade_count > 0 else 0.0
    
    def is_toxic(self, threshold: float = 0.5) -> bool:
        """Check if client appears to be informed/toxic."""
        # Low fill ratio + high edge = toxic
        if self.rfq_count < 5:
            return False  # Not enough data
        
        return self.get_fill_ratio() < 0.3 and abs(self.get_avg_edge()) > threshold


class Client(ABC):
    """
    Base class for all client types.
    
    Clients generate RFQs and decide whether to trade based on dealer quotes.
    """
    
    def __init__(
        self,
        client_id: str,
        name: str,
        client_type: str,
        rfq_probability: float = 0.1,
        mean_size: float = 1.0,
        size_std: float = 0.5,
        sector_preferences: Optional[dict] = None,
        buy_sell_skew: float = 0.0
    ):
        """
        Args:
            client_id: Unique identifier
            name: Human-readable name
            client_type: Type classification
            rfq_probability: Probability of sending RFQ per timestep
            mean_size: Average trade size
            size_std: Standard deviation of trade size
            sector_preferences: Dict mapping sector to preference weight
            buy_sell_skew: -1.0 (all sells) to +1.0 (all buys), 0.0 = neutral
        """
        self.client_id = client_id
        self.name = name
        self.client_type = client_type
        self.rfq_probability = rfq_probability
        self.mean_size = mean_size
        self.size_std = size_std
        self.sector_preferences = sector_preferences or {}
        self.buy_sell_skew = buy_sell_skew
        
        # Internal state
        self._rfq_counter = 0
    
    def maybe_generate_rfq(
        self,
        timestep: int,
        market_state: MarketState,
        bonds: List[Bond]
    ) -> Optional[RFQ]:
        """
        Maybe generate an RFQ this timestep.
        
        Returns:
            RFQ if generated, None otherwise
        """
        if random.random() > self.rfq_probability:
            return None
        
        # Select a bond
        bond = self._select_bond(bonds)
        if bond is None:
            return None
        
        # Determine side and size
        # Apply buy_sell_skew: -1 = all sells, +1 = all buys, 0 = neutral
        if self.buy_sell_skew == 0.0:
            side = random.choice(["buy", "sell"])
        else:
            # Convert skew to probability of buy
            # skew = -1 -> p_buy = 0
            # skew = 0 -> p_buy = 0.5
            # skew = 1 -> p_buy = 1
            p_buy = (self.buy_sell_skew + 1.0) / 2.0
            side = "buy" if random.random() < p_buy else "sell"
        
        size = max(0.1, random.gauss(self.mean_size, self.size_std))
        
        # Determine if this is fishing
        is_fishing = self._is_fishing_rfq()
        
        self._rfq_counter += 1
        rfq_id = f"{self.client_id}_RFQ{self._rfq_counter:04d}"
        
        return RFQ(
            rfq_id=rfq_id,
            timestamp=timestep,
            client_id=self.client_id,
            bond_id=bond.id,
            side=side,
            size=round(size, 2),
            is_fishing=is_fishing
        )
    
    def _select_bond(self, bonds: List[Bond]) -> Optional[Bond]:
        """Select a bond to request quote for."""
        if not bonds:
            return None
        
        # Apply sector preferences if any
        if self.sector_preferences:
            weights = []
            for bond in bonds:
                weight = self.sector_preferences.get(bond.sector, 1.0)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                return random.choices(bonds, weights=weights, k=1)[0]
        
        return random.choice(bonds)
    
    @abstractmethod
    def _is_fishing_rfq(self) -> bool:
        """Determine if this RFQ is just fishing for price."""
        pass
    
    @abstractmethod
    def decide_trade(
        self,
        rfq: RFQ,
        quote: Quote,
        fair_value_estimate: float
    ) -> bool:
        """
        Decide whether to accept the dealer's quote.
        
        Args:
            rfq: The original RFQ
            quote: Dealer's quote
            fair_value_estimate: Client's estimate of fair value
            
        Returns:
            True to trade, False to pass
        """
        pass
    
    def get_fair_value_estimate(self, bond: Bond, market_state: MarketState) -> float:
        """
        Get client's estimate of fair value.
        
        Different clients have different information quality.
        Base implementation: very noisy estimate.
        """
        true_fair = bond.get_true_fair_price()
        noise = random.gauss(0, bond.volatility * 10)  # Large noise
        return true_fair + noise


class RealMoneyClient(Client):
    """
    Real money client (asset manager, pension fund).
    
    - Low RFQ frequency
    - Large sizes
    - Rarely fishes
    - Trades if price is reasonable
    """
    
    def __init__(self, client_id: str, name: str):
        super().__init__(
            client_id=client_id,
            name=name,
            client_type="real_money",
            rfq_probability=0.05,
            mean_size=3.0,
            size_std=1.0
        )
        self.fishing_probability = 0.05
        self.trade_threshold = 1.0  # Trade if within 1.0 of fair value
    
    def _is_fishing_rfq(self) -> bool:
        return random.random() < self.fishing_probability
    
    def decide_trade(self, rfq: RFQ, quote: Quote, fair_value_estimate: float) -> bool:
        # Don't trade if just fishing
        if rfq.is_fishing:
            return False
        
        # Trade if quote is reasonable
        if rfq.side == "buy":
            # Client buying - wants price not too high
            return quote.price <= fair_value_estimate + self.trade_threshold
        else:
            # Client selling - wants price not too low
            return quote.price >= fair_value_estimate - self.trade_threshold


class HedgeFundClient(Client):
    """
    Hedge fund client - often informed.
    
    - Higher RFQ frequency
    - More accurate fair value estimates
    - Trades aggressively when quote is favorable
    """
    
    def __init__(self, client_id: str, name: str):
        super().__init__(
            client_id=client_id,
            name=name,
            client_type="hedge_fund",
            rfq_probability=0.15,
            mean_size=2.0,
            size_std=1.0
        )
        self.fishing_probability = 0.20
        self.information_quality = 0.8  # Better info
    
    def _is_fishing_rfq(self) -> bool:
        return random.random() < self.fishing_probability
    
    def get_fair_value_estimate(self, bond: Bond, market_state: MarketState) -> float:
        """Hedge funds have better information."""
        true_fair = bond.get_true_fair_price()
        # Much less noise due to better research
        noise = random.gauss(0, bond.volatility * 3 * (1 - self.information_quality))
        return true_fair + noise
    
    def decide_trade(self, rfq: RFQ, quote: Quote, fair_value_estimate: float) -> bool:
        if rfq.is_fishing:
            return False
        
        # Trade if we have an edge
        if rfq.side == "buy":
            # We're buying - good if quote < fair (we think it's cheap)
            return quote.price < fair_value_estimate
        else:
            # We're selling - good if quote > fair (we think it's expensive)
            return quote.price > fair_value_estimate


class FisherClient(Client):
    """
    Fisher client - primarily seeking information.
    
    - High RFQ frequency
    - Most RFQs are fishing
    - Almost never trades unless quote is extremely favorable
    """
    
    def __init__(self, client_id: str, name: str):
        super().__init__(
            client_id=client_id,
            name=name,
            client_type="fisher",
            rfq_probability=0.25,
            mean_size=1.5,
            size_std=0.5
        )
        self.fishing_probability = 0.80
    
    def _is_fishing_rfq(self) -> bool:
        return random.random() < self.fishing_probability
    
    def decide_trade(self, rfq: RFQ, quote: Quote, fair_value_estimate: float) -> bool:
        if rfq.is_fishing:
            return False
        
        # Only trade if quote is VERY favorable (2+ edge)
        edge_threshold = 2.0
        
        if rfq.side == "buy":
            return quote.price < fair_value_estimate - edge_threshold
        else:
            return quote.price > fair_value_estimate + edge_threshold


class NoiseClient(Client):
    """
    Noise trader - random behavior.
    
    - Random RFQ generation
    - Random trade decisions
    - No sophistication
    """
    
    def __init__(self, client_id: str, name: str):
        super().__init__(
            client_id=client_id,
            name=name,
            client_type="noise",
            rfq_probability=0.10,
            mean_size=1.0,
            size_std=0.8
        )
        self.fishing_probability = 0.30
        self.trade_probability = 0.40  # Random trade probability
    
    def _is_fishing_rfq(self) -> bool:
        return random.random() < self.fishing_probability
    
    def decide_trade(self, rfq: RFQ, quote: Quote, fair_value_estimate: float) -> bool:
        if rfq.is_fishing:
            return False
        
        # Just random
        return random.random() < self.trade_probability


def create_client_universe(
    num_real_money: int = 3,
    num_hedge_fund: int = 2,
    num_fisher: int = 2,
    num_noise: int = 3
) -> List[Client]:
    """Create a universe of clients with different types."""
    clients = []
    
    for i in range(num_real_money):
        clients.append(RealMoneyClient(
            client_id=f"RM{i+1:02d}",
            name=f"Real Money Fund {i+1}"
        ))
    
    for i in range(num_hedge_fund):
        clients.append(HedgeFundClient(
            client_id=f"HF{i+1:02d}",
            name=f"Hedge Fund {i+1}"
        ))
    
    for i in range(num_fisher):
        clients.append(FisherClient(
            client_id=f"FI{i+1:02d}",
            name=f"Fisher {i+1}"
        ))
    
    for i in range(num_noise):
        clients.append(NoiseClient(
            client_id=f"NO{i+1:02d}",
            name=f"Noise Trader {i+1}"
        ))
    
    return clients


def create_clients_from_scenario(
    client_counts: dict,
    rfq_probability_multiplier: float = 1.0,
    size_multiplier: float = 1.0
) -> List[Client]:
    """
    Create a universe of clients based on scenario specifications.
    
    Args:
        client_counts: Dictionary with counts per type (from ScenarioConfig.get_total_client_count())
        rfq_probability_multiplier: Scale all client RFQ probabilities
        size_multiplier: Scale all client trade sizes
        
    Returns:
        List of Client objects
    """
    clients = []
    
    # Real money clients
    num_rm = client_counts.get('real_money', 0)
    for i in range(num_rm):
        client = RealMoneyClient(
            client_id=f"RM{i+1:02d}",
            name=f"Real Money Fund {i+1}"
        )
        client.rfq_probability *= rfq_probability_multiplier
        client.mean_size *= size_multiplier
        clients.append(client)
    
    # Hedge fund clients
    num_hf = client_counts.get('hedge_fund', 0)
    for i in range(num_hf):
        client = HedgeFundClient(
            client_id=f"HF{i+1:02d}",
            name=f"Hedge Fund {i+1}"
        )
        client.rfq_probability *= rfq_probability_multiplier
        client.mean_size *= size_multiplier
        clients.append(client)
    
    # Fisher clients
    num_fisher = client_counts.get('fisher', 0)
    for i in range(num_fisher):
        client = FisherClient(
            client_id=f"FI{i+1:02d}",
            name=f"Fisher {i+1}"
        )
        client.rfq_probability *= rfq_probability_multiplier
        client.mean_size *= size_multiplier
        clients.append(client)
    
    # Noise clients
    num_noise = client_counts.get('noise', 0)
    for i in range(num_noise):
        client = NoiseClient(
            client_id=f"NO{i+1:02d}",
            name=f"Noise Trader {i+1}"
        )
        client.rfq_probability *= rfq_probability_multiplier
        client.mean_size *= size_multiplier
        clients.append(client)
    
    return clients
