"""
Market state and dynamics including impact model.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from illiquid_market_sim.bonds import Bond, get_bonds_by_issuer, get_bonds_by_sector


@dataclass
class MarketState:
    """
    Represents the current state of the market including factor levels.
    
    The market has common factors that drive bond prices:
    - Sector levels (IG, HY, EM)
    - Occasionally experiences jumps (credit events)
    
    Bonds' true fair prices are updated based on these factors.
    """
    
    current_step: int = 0
    
    # Factor levels (changes in spread, in bps)
    level_IG: float = 0.0
    level_HY: float = 0.0
    level_EM: float = 0.0
    
    # Market parameters
    volatility: float = 0.02
    jump_probability: float = 0.05
    
    # History
    _factor_history: List[Dict[str, float]] = field(default_factory=list, repr=False)
    
    def get_factors(self) -> Dict[str, float]:
        """Get current factor levels."""
        return {
            "level_IG": self.level_IG,
            "level_HY": self.level_HY,
            "level_EM": self.level_EM,
        }
    
    def step(self, bonds: List[Bond]) -> Optional[Dict[str, any]]:
        """
        Advance market state by one timestep.
        
        - Evolve factors with Brownian motion
        - Occasionally trigger jump events (credit events)
        - Update bond fair prices accordingly
        
        Returns:
            Event information if a jump occurred, None otherwise
        """
        self.current_step += 1
        
        # Store current state
        self._factor_history.append(self.get_factors())
        
        # Evolve factors with correlated Brownian motion
        ig_shock = random.gauss(0, self.volatility)
        hy_shock = random.gauss(0, self.volatility * 1.5)  # More volatile
        em_shock = random.gauss(0, self.volatility * 1.8)
        
        self.level_IG += ig_shock
        self.level_HY += hy_shock
        self.level_EM += em_shock
        
        # Update all bond fair prices based on factors
        for bond in bonds:
            # Base update from sector factor
            sector_change = self.get_factors()[f"level_{bond.sector}"]
            new_price = 100 + sector_change
            
            # Add bond-specific noise
            bond_noise = random.gauss(0, bond.volatility)
            new_price += bond_noise
            
            bond.update_fair_price(new_price)
        
        # Check for jump events (credit events)
        event = None
        if random.random() < self.jump_probability:
            event = self._trigger_jump_event(bonds)
        
        return event
    
    def _trigger_jump_event(self, bonds: List[Bond]) -> Dict[str, any]:
        """
        Trigger a jump event (credit event, sector shock, etc.).
        """
        event_type = random.choice(["sector_shock", "issuer_downgrade"])
        
        if event_type == "sector_shock":
            # Pick a sector and apply a large shock
            sector = random.choice(["IG", "HY", "EM"])
            shock_size = random.uniform(-5, -2)  # Negative shock (spreads widen)
            
            # Apply to sector factor
            if sector == "IG":
                self.level_IG += shock_size
            elif sector == "HY":
                self.level_HY += shock_size
            else:
                self.level_EM += shock_size
            
            # Apply to all bonds in that sector
            sector_bonds = get_bonds_by_sector(bonds, sector)
            for bond in sector_bonds:
                bond.apply_impact(shock_size)
            
            return {
                "type": "sector_shock",
                "sector": sector,
                "shock": shock_size,
                "affected_bonds": len(sector_bonds)
            }
        
        else:  # issuer_downgrade
            # Pick a random issuer
            issuers = list(set(b.issuer for b in bonds))
            issuer = random.choice(issuers)
            shock_size = random.uniform(-10, -3)  # Larger shock for single issuer
            
            # Apply to all bonds of that issuer
            issuer_bonds = get_bonds_by_issuer(bonds, issuer)
            for bond in issuer_bonds:
                bond.apply_impact(shock_size)
            
            return {
                "type": "issuer_downgrade",
                "issuer": issuer,
                "shock": shock_size,
                "affected_bonds": len(issuer_bonds)
            }
    
    def get_fair_price(self, bond: Bond) -> float:
        """
        Get the true fair price for a bond (hidden from dealer).
        """
        return bond.get_true_fair_price()


class MarketImpactModel:
    """
    Models the impact of trades on bond prices.
    
    When a trade occurs:
    1. The traded bond's price moves based on size and liquidity
    2. Related bonds (same issuer/sector) also move (cross-impact)
    """
    
    def __init__(
        self,
        base_impact_coeff: float = 0.001,
        cross_impact_factor: float = 0.3,
        impact_decay: float = 0.5
    ):
        """
        Args:
            base_impact_coeff: Base impact per unit size
            cross_impact_factor: Fraction of impact that spills to related bonds
            impact_decay: Not used yet, for future: how impact decays over time
        """
        self.base_impact_coeff = base_impact_coeff
        self.cross_impact_factor = cross_impact_factor
        self.impact_decay = impact_decay
    
    def apply_trade_impact(
        self,
        traded_bond: Bond,
        side: Literal["buy", "sell"],
        size: float,
        all_bonds: List[Bond]
    ) -> Dict[str, float]:
        """
        Apply market impact from a trade.
        
        Args:
            traded_bond: The bond that was traded
            side: Dealer's side ("buy" = dealer bought, price moves up)
            size: Trade size
            all_bonds: All bonds in the universe
            
        Returns:
            Dictionary mapping bond_id to impact applied
        """
        impacts = {}
        
        # Direct impact on traded bond
        # If dealer buys, price moves up (positive impact)
        # If dealer sells, price moves down (negative impact)
        direction = 1.0 if side == "buy" else -1.0
        
        # Impact inversely proportional to liquidity
        liquidity_factor = max(traded_bond.liquidity, 0.01)  # Avoid division by zero
        direct_impact = (
            direction * 
            self.base_impact_coeff * 
            size / liquidity_factor
        )
        
        traded_bond.apply_impact(direct_impact)
        impacts[traded_bond.id] = direct_impact
        
        # Cross-impact on related bonds
        cross_impact = direct_impact * self.cross_impact_factor
        
        for bond in all_bonds:
            if bond.id == traded_bond.id:
                continue
            
            # Impact same issuer more
            if bond.issuer == traded_bond.issuer:
                impact = cross_impact * 0.8
                bond.apply_impact(impact)
                impacts[bond.id] = impact
            
            # Impact same sector less
            elif bond.sector == traded_bond.sector:
                impact = cross_impact * 0.3
                bond.apply_impact(impact)
                impacts[bond.id] = impact
        
        return impacts
