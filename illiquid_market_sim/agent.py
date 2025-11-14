"""
Dealer agent for quoting RFQs.

This module contains the dealer's quoting logic. Currently rule-based,
but structured to allow plugging in ML models later.
"""

from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass

from illiquid_market_sim.bonds import Bond
from illiquid_market_sim.rfq import RFQ, Quote
from illiquid_market_sim.portfolio import Portfolio
from illiquid_market_sim.market import MarketState
from illiquid_market_sim.clients import ClientStats


class QuotingStrategy(Protocol):
    """Protocol for quoting strategies (allows plugging in different strategies)."""
    
    def generate_quote(
        self,
        rfq: RFQ,
        bond: Bond,
        portfolio: Portfolio,
        market_state: MarketState,
        client_stats: Optional[ClientStats]
    ) -> Quote:
        """Generate a quote for an RFQ."""
        ...


@dataclass
class RuleBasedQuotingStrategy:
    """
    Simple rule-based quoting strategy.
    
    Quotes are based on:
    1. Naive mid estimate (from bond observables)
    2. Base spread
    3. Liquidity adjustment
    4. Inventory risk adjustment
    5. Client toxicity adjustment
    """
    
    base_spread_bps: float = 50.0
    illiquidity_spread_multiplier: float = 2.0
    inventory_risk_penalty: float = 0.01
    max_inventory_per_bond: float = 10.0
    toxic_client_spread_multiplier: float = 1.5
    
    def generate_quote(
        self,
        rfq: RFQ,
        bond: Bond,
        portfolio: Portfolio,
        market_state: MarketState,
        client_stats: Optional[ClientStats]
    ) -> Quote:
        """Generate a quote using rule-based logic."""
        
        # 1. Get naive mid estimate
        naive_mid = self._estimate_mid(bond, portfolio, market_state)
        
        # 2. Calculate base spread
        spread_bps = self.base_spread_bps
        
        # 3. Adjust for liquidity
        liquidity_mult = 1.0 + (1.0 - bond.liquidity) * self.illiquidity_spread_multiplier
        spread_bps *= liquidity_mult
        
        # 4. Adjust for inventory risk
        position = portfolio.get_position(bond.id)
        inventory = position.quantity
        
        # If we're long and client wants to sell (we'd buy more), widen spread / lower bid
        # If we're short and client wants to buy (we'd sell more), widen spread / raise offer
        inventory_adjustment = 0.0
        if rfq.side == "buy":
            # Client buying, we're selling
            # If we're already short, we want to be paid more to sell more
            inventory_adjustment = -inventory * self.inventory_risk_penalty
        else:
            # Client selling, we're buying
            # If we're already long, we want to pay less to buy more
            inventory_adjustment = inventory * self.inventory_risk_penalty
        
        # 5. Adjust for client behavior
        if client_stats and client_stats.is_toxic():
            spread_bps *= self.toxic_client_spread_multiplier
        
        # 6. Calculate final quote price
        spread_pts = spread_bps / 100.0  # Convert bps to points
        
        if rfq.side == "buy":
            # Client buying, we're selling -> offer side
            # Quote above mid
            price = naive_mid + spread_pts / 2 + inventory_adjustment
        else:
            # Client selling, we're buying -> bid side
            # Quote below mid
            price = naive_mid - spread_pts / 2 + inventory_adjustment
        
        # 7. Check inventory limits - quote very wide if at limit
        if abs(inventory) > self.max_inventory_per_bond:
            if rfq.side == "buy" and inventory < 0:
                # Already too short, don't want to sell more
                price += 5.0  # Quote very high
            elif rfq.side == "sell" and inventory > 0:
                # Already too long, don't want to buy more
                price -= 5.0  # Quote very low
        
        return Quote(
            rfq_id=rfq.rfq_id,
            price=round(price, 2),
            spread_bps=round(spread_bps, 1),
            timestamp=rfq.timestamp
        )
    
    def _estimate_mid(
        self,
        bond: Bond,
        portfolio: Portfolio,
        market_state: MarketState
    ) -> float:
        """
        Estimate the mid price for a bond.
        
        This is the dealer's internal view, which is imperfect.
        Uses last traded price if available, otherwise a model-based estimate.
        """
        # Use last traded price if we have it
        last_price = bond.get_last_traded_price()
        if last_price is not None:
            return last_price
        
        # Otherwise, use naive mid from bond's method
        return bond.get_naive_mid(market_state.get_factors())


class DealerAgent:
    """
    The dealer agent that manages quoting and position taking.
    
    Attributes:
        portfolio: Current positions
        quoting_strategy: Strategy for generating quotes
        client_stats: Statistics per client
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        quoting_strategy: Optional[QuotingStrategy] = None
    ):
        """
        Args:
            portfolio: Portfolio to manage
            quoting_strategy: Strategy to use for quoting (defaults to rule-based)
        """
        self.portfolio = portfolio
        self.quoting_strategy = quoting_strategy or RuleBasedQuotingStrategy()
        self.client_stats: Dict[str, ClientStats] = {}
    
    def quote_for_rfq(
        self,
        rfq: RFQ,
        bond: Bond,
        market_state: MarketState
    ) -> Quote:
        """
        Generate a quote for an RFQ.
        
        Args:
            rfq: The RFQ to quote
            bond: The bond being quoted
            market_state: Current market state
            
        Returns:
            Quote object
        """
        # Get or create client stats
        if rfq.client_id not in self.client_stats:
            self.client_stats[rfq.client_id] = ClientStats(client_id=rfq.client_id)
        
        client_stats = self.client_stats[rfq.client_id]
        client_stats.rfq_count += 1
        
        # Generate quote using strategy
        quote = self.quoting_strategy.generate_quote(
            rfq=rfq,
            bond=bond,
            portfolio=self.portfolio,
            market_state=market_state,
            client_stats=client_stats
        )
        
        return quote
    
    def record_trade(
        self,
        rfq: RFQ,
        quote: Quote,
        fair_value: float
    ) -> None:
        """
        Record that a trade occurred (for client stats tracking).
        
        Args:
            rfq: The RFQ that was filled
            quote: The quote that was accepted
            fair_value: True fair value at time of trade (for edge calculation)
        """
        client_stats = self.client_stats.get(rfq.client_id)
        if client_stats:
            client_stats.trade_count += 1
            
            # Calculate edge captured by client
            # If client bought at quote price below fair, they captured edge
            # If client sold at quote price above fair, they captured edge
            if rfq.side == "buy":
                # Client bought, dealer sold
                # Client edge = fair - price (positive if bought below fair)
                edge = fair_value - quote.price
            else:
                # Client sold, dealer bought
                # Client edge = price - fair (positive if sold above fair)
                edge = quote.price - fair_value
            
            client_stats.total_edge_captured += edge
    
    def get_client_stats(self, client_id: str) -> Optional[ClientStats]:
        """Get statistics for a specific client."""
        return self.client_stats.get(client_id)
    
    def get_all_client_stats(self) -> Dict[str, ClientStats]:
        """Get statistics for all clients."""
        return self.client_stats.copy()


class ManualQuotingStrategy:
    """
    Manual quoting strategy for human-in-the-loop game mode.
    
    This strategy prompts the user for a quote price.
    """
    
    def __init__(self, naive_spread_bps: float = 50.0):
        self.naive_spread_bps = naive_spread_bps
    
    def generate_quote(
        self,
        rfq: RFQ,
        bond: Bond,
        portfolio: Portfolio,
        market_state: MarketState,
        client_stats: Optional[ClientStats]
    ) -> Quote:
        """Prompt user for quote price."""
        
        # Calculate naive mid for reference
        last_price = bond.get_last_traded_price()
        if last_price is not None:
            naive_mid = last_price
        else:
            naive_mid = bond.get_naive_mid(market_state.get_factors())
        
        # Display RFQ info
        print(f"\n{'='*60}")
        print(f"RFQ: {rfq}")
        print(f"Bond: {bond.id} ({bond.issuer}, {bond.sector}, {bond.rating})")
        print(f"Maturity: {bond.maturity_years:.1f}y, Liquidity: {bond.liquidity:.2f}")
        print(f"Naive mid estimate: {naive_mid:.2f}")
        
        # Show current position
        position = portfolio.get_position(bond.id)
        print(f"Current position: {position.quantity:.2f}")
        
        # Show client stats if available
        if client_stats and client_stats.rfq_count > 0:
            print(f"Client stats: {client_stats.rfq_count} RFQs, "
                  f"{client_stats.trade_count} trades, "
                  f"fill ratio: {client_stats.get_fill_ratio():.1%}")
        
        print(f"{'='*60}")
        
        # Get user input
        while True:
            try:
                price_str = input(f"Enter your quote price (or 'pass' to skip): ").strip()
                if price_str.lower() in ['pass', 'p', 'skip', 's']:
                    # Quote extremely wide to ensure no trade
                    price = 0.01 if rfq.side == "sell" else 999.0
                    break
                
                price = float(price_str)
                if price <= 0:
                    print("Price must be positive")
                    continue
                break
            except ValueError:
                print("Invalid input, please enter a number or 'pass'")
        
        spread_bps = abs(price - naive_mid) * 100 * 2  # Rough estimate
        
        return Quote(
            rfq_id=rfq.rfq_id,
            price=round(price, 2),
            spread_bps=round(spread_bps, 1),
            timestamp=rfq.timestamp
        )
