"""
Portfolio and position tracking for the dealer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from illiquid_market_sim.bonds import Bond


@dataclass
class Position:
    """
    Represents a position in a single bond.
    
    Attributes:
        bond_id: Bond identifier
        quantity: Current position (positive = long, negative = short)
        total_cost: Total cost basis (sum of all purchase costs)
        trades: Number of trades in this position
    """
    
    bond_id: str
    quantity: float = 0.0
    total_cost: float = 0.0
    trades: int = 0
    
    def get_average_cost(self) -> Optional[float]:
        """Get average cost per unit, if position exists."""
        if abs(self.quantity) < 1e-6:
            return None
        return self.total_cost / self.quantity if self.quantity != 0 else None
    
    def update_trade(self, side: Literal["buy", "sell"], size: float, price: float) -> None:
        """
        Update position based on a trade.
        
        Args:
            side: "buy" or "sell"
            size: Quantity traded
            price: Execution price
        """
        old_quantity = self.quantity
        
        if side == "buy":
            new_quantity = old_quantity + size
        else:  # sell
            new_quantity = old_quantity - size
        
        # Check if position crosses zero (reverses direction)
        crosses_zero = (old_quantity > 0 and new_quantity < 0) or (old_quantity < 0 and new_quantity > 0)
        
        if crosses_zero:
            # Position is reversing - need to reset cost basis for new position
            # The remaining size after closing the old position
            remaining_size = abs(new_quantity)
            
            # Set new cost basis for the new position
            self.quantity = new_quantity
            self.total_cost = remaining_size * price
        elif abs(new_quantity) < 1e-6:
            # Position goes to exactly zero - reset everything
            self.quantity = 0.0
            self.total_cost = 0.0
        else:
            # Normal case - adding to or reducing position without crossing zero
            if side == "buy":
                self.quantity += size
                self.total_cost += size * price
            else:  # sell
                self.quantity -= size
                self.total_cost -= size * price
        
        self.trades += 1
    
    def get_market_value(self, current_price: float) -> float:
        """Get current market value of position."""
        return self.quantity * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Get unrealized P&L."""
        market_value = self.get_market_value(current_price)
        return market_value - self.total_cost


@dataclass
class Portfolio:
    """
    Manages all dealer positions and P&L tracking.
    """
    
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # P&L tracking
    realized_pnl: float = 0.0
    total_trading_fees: float = 0.0
    
    # History
    _pnl_history: List[Dict[str, float]] = field(default_factory=list, repr=False)
    
    def get_position(self, bond_id: str) -> Position:
        """Get position for a bond (creates if doesn't exist)."""
        if bond_id not in self.positions:
            self.positions[bond_id] = Position(bond_id=bond_id)
        return self.positions[bond_id]
    
    def update_on_trade(
        self,
        bond_id: str,
        side: Literal["buy", "sell"],
        size: float,
        price: float
    ) -> None:
        """
        Update portfolio based on a trade.
        
        Args:
            bond_id: Bond traded
            side: Dealer's side
            size: Quantity
            price: Execution price
        """
        position = self.get_position(bond_id)
        
        # For realized P&L tracking, check if we're reducing a position
        old_quantity = position.quantity
        old_avg_cost = position.get_average_cost()
        
        # Update the position
        position.update_trade(side, size, price)
        
        # Calculate realized P&L if position was reduced
        if old_avg_cost is not None:
            if side == "sell" and old_quantity > 0:
                # Selling from a long position
                size_reduced = min(size, old_quantity)
                realized = size_reduced * (price - old_avg_cost)
                self.realized_pnl += realized
            elif side == "buy" and old_quantity < 0:
                # Buying to cover a short position
                size_reduced = min(size, abs(old_quantity))
                # For short positions, old_avg_cost is negative, so use absolute value
                realized = size_reduced * (abs(old_avg_cost) - price)
                self.realized_pnl += realized
    
    def mark_to_market(self, bonds: List[Bond]) -> Dict[str, float]:
        """
        Mark all positions to market.
        
        Args:
            bonds: List of all bonds (to get current prices)
            
        Returns:
            Dictionary with MTM values and P&L breakdown
        """
        bond_dict = {b.id: b for b in bonds}
        
        total_market_value = 0.0
        total_cost_basis = 0.0
        unrealized_pnl = 0.0
        
        position_details = {}
        
        for bond_id, position in self.positions.items():
            if abs(position.quantity) < 1e-6:
                continue
            
            bond = bond_dict.get(bond_id)
            if bond is None:
                continue
            
            # Use naive mid as the marking price (dealer doesn't know true fair value)
            # In a real system, this would be the dealer's internal mark
            # For now, use last traded price or a fallback
            current_price = bond.get_last_traded_price()
            if current_price is None:
                current_price = bond.get_true_fair_price()  # Fallback
            
            market_value = position.get_market_value(current_price)
            pos_unrealized = position.get_unrealized_pnl(current_price)
            
            total_market_value += market_value
            total_cost_basis += position.total_cost
            unrealized_pnl += pos_unrealized
            
            position_details[bond_id] = {
                "quantity": position.quantity,
                "avg_cost": position.get_average_cost(),
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": pos_unrealized
            }
        
        mtm_result = {
            "total_market_value": total_market_value,
            "total_cost_basis": total_cost_basis,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": unrealized_pnl + self.realized_pnl,
            "positions": position_details
        }
        
        # Store in history
        self._pnl_history.append({
            "unrealized": unrealized_pnl,
            "realized": self.realized_pnl,
            "total": unrealized_pnl + self.realized_pnl,
            "market_value": total_market_value
        })
        
        return mtm_result
    
    def get_inventory_risk(self) -> float:
        """
        Calculate a simple inventory risk metric.
        Sum of absolute values of all positions.
        """
        return sum(abs(pos.quantity) for pos in self.positions.values())
    
    def get_pnl_history(self) -> List[Dict[str, float]]:
        """Get historical P&L snapshots."""
        return self._pnl_history
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics of the portfolio."""
        num_positions = sum(1 for pos in self.positions.values() if abs(pos.quantity) > 1e-6)
        total_trades = sum(pos.trades for pos in self.positions.values())
        inventory = self.get_inventory_risk()
        
        return {
            "num_positions": num_positions,
            "total_trades": total_trades,
            "inventory_risk": inventory,
            "realized_pnl": self.realized_pnl
        }
