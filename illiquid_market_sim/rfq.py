"""
Request for Quote (RFQ) model.
"""

from dataclasses import dataclass
from typing import Literal
from datetime import datetime


@dataclass
class RFQ:
    """
    Represents a Request for Quote from a client.
    
    Attributes:
        rfq_id: Unique identifier for this RFQ
        timestamp: When the RFQ was created (step number or datetime)
        client_id: ID of the client sending the RFQ
        bond_id: Bond being quoted
        side: Whether client wants to buy or sell
        size: Quantity requested
        is_fishing: Hidden truth - is this just price discovery? (not observable by dealer)
    """
    
    rfq_id: str
    timestamp: int  # step number
    client_id: str
    bond_id: str
    side: Literal["buy", "sell"]
    size: float
    is_fishing: bool = False
    
    def __str__(self) -> str:
        """Human-readable representation of the RFQ."""
        return (f"RFQ[{self.rfq_id}] {self.client_id} wants to {self.side.upper()} "
                f"{self.size:.2f} of {self.bond_id} @ t={self.timestamp}")
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Quote:
    """
    Dealer's quote in response to an RFQ.
    
    Attributes:
        rfq_id: Reference to the RFQ
        price: Quoted price
        spread_bps: Spread in basis points used
        timestamp: When quote was generated
    """
    
    rfq_id: str
    price: float
    spread_bps: float
    timestamp: int
    
    def __str__(self) -> str:
        return f"Quote[{self.rfq_id}] price={self.price:.2f}, spread={self.spread_bps:.1f}bps"


@dataclass
class Trade:
    """
    Executed trade resulting from an accepted quote.
    
    Attributes:
        trade_id: Unique trade identifier
        rfq_id: Original RFQ that led to this trade
        client_id: Client counterparty
        bond_id: Bond traded
        side: Dealer's side (opposite of client's RFQ side)
        size: Quantity traded
        price: Execution price
        timestamp: When trade executed
    """
    
    trade_id: str
    rfq_id: str
    client_id: str
    bond_id: str
    side: Literal["buy", "sell"]  # Dealer's side
    size: float
    price: float
    timestamp: int
    
    def __str__(self) -> str:
        return (f"Trade[{self.trade_id}] {self.bond_id}: "
                f"dealer {self.side} {self.size:.2f} @ {self.price:.2f}")
    
    @classmethod
    def from_rfq_and_quote(cls, rfq: RFQ, quote: Quote, trade_id: str) -> "Trade":
        """Create a trade from an RFQ and accepted quote."""
        # Dealer's side is opposite of client's
        dealer_side = "sell" if rfq.side == "buy" else "buy"
        
        return cls(
            trade_id=trade_id,
            rfq_id=rfq.rfq_id,
            client_id=rfq.client_id,
            bond_id=rfq.bond_id,
            side=dealer_side,
            size=rfq.size,
            price=quote.price,
            timestamp=quote.timestamp
        )
