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
    spread_drift_bps: float = 0.0  # Persistent drift per step
    sector_shock_prob: float = 0.05
    idiosyncratic_event_prob: float = 0.03
    
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
        
        - Evolve factors with Brownian motion and drift
        - Occasionally trigger jump events (credit events)
        - Update bond fair prices accordingly
        
        Returns:
            Event information if a jump occurred, None otherwise
        """
        self.current_step += 1
        
        # Store current state
        self._factor_history.append(self.get_factors())
        
        # Evolve factors with correlated Brownian motion + drift
        ig_shock = random.gauss(0, self.volatility) + self.spread_drift_bps
        hy_shock = random.gauss(0, self.volatility * 1.5) + self.spread_drift_bps * 1.2  # HY drifts more
        em_shock = random.gauss(0, self.volatility * 1.8) + self.spread_drift_bps * 1.3
        
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
        
        # Check for different types of jump events
        event = None
        
        # Market-wide shock
        if random.random() < self.jump_probability:
            event = self._trigger_market_shock(bonds)
        # Sector-specific shock
        elif random.random() < self.sector_shock_prob:
            event = self._trigger_sector_shock(bonds)
        # Idiosyncratic issuer event
        elif random.random() < self.idiosyncratic_event_prob:
            event = self._trigger_issuer_event(bonds)
        
        return event
    
    def _trigger_market_shock(self, bonds: List[Bond]) -> Dict[str, any]:
        """Trigger a broad market shock affecting all sectors."""
        shock_size = random.uniform(-8, -3)  # Negative = spreads widen
        
        # Apply to all sector factors
        self.level_IG += shock_size * 0.8
        self.level_HY += shock_size * 1.2
        self.level_EM += shock_size * 1.4
        
        # Apply to all bonds
        for bond in bonds:
            sector_multiplier = {"IG": 0.8, "HY": 1.2, "EM": 1.4}[bond.sector]
            bond.apply_impact(shock_size * sector_multiplier)
        
        return {
            "type": "market_shock",
            "shock": shock_size,
            "affected_bonds": len(bonds)
        }
    
    def _trigger_sector_shock(self, bonds: List[Bond]) -> Dict[str, any]:
        """Trigger a sector-specific shock."""
        sector = random.choice(["IG", "HY", "EM"])
        shock_size = random.uniform(-6, -2)  # Negative shock (spreads widen)
        
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
    
    def _trigger_issuer_event(self, bonds: List[Bond]) -> Dict[str, any]:
        """Trigger an idiosyncratic issuer event (downgrade, rumor, etc.)."""
        # Pick a random issuer
        issuers = list(set(b.issuer for b in bonds))
        issuer = random.choice(issuers)
        shock_size = random.uniform(-12, -3)  # Large shock for single issuer
        
        # Apply to all bonds of that issuer
        issuer_bonds = get_bonds_by_issuer(bonds, issuer)
        for bond in issuer_bonds:
            bond.apply_impact(shock_size)
        
        return {
            "type": "issuer_event",
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
        impact_decay: float = 0.5,
        liquidity_multiplier: float = 1.0,
        impact_cross_issuer: float = 0.3,
        impact_cross_sector: float = 0.1
    ):
        """
        Args:
            base_impact_coeff: Base impact per unit size
            cross_impact_factor: Fraction of impact that spills to related bonds (legacy)
            impact_decay: Not used yet, for future: how impact decays over time
            liquidity_multiplier: Multiplier for liquidity (>1 = worse liquidity)
            impact_cross_issuer: Fraction of impact spilling to same issuer
            impact_cross_sector: Fraction of impact spilling to same sector
        """
        self.base_impact_coeff = base_impact_coeff
        self.cross_impact_factor = cross_impact_factor
        self.impact_decay = impact_decay
        self.liquidity_multiplier = liquidity_multiplier
        self.impact_cross_issuer = impact_cross_issuer
        self.impact_cross_sector = impact_cross_sector
    
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
        
        # Impact inversely proportional to liquidity, scaled by liquidity multiplier
        liquidity_factor = max(traded_bond.liquidity, 0.01)  # Avoid division by zero
        effective_liquidity = liquidity_factor / self.liquidity_multiplier
        
        direct_impact = (
            direction * 
            self.base_impact_coeff * 
            size / effective_liquidity
        )
        
        traded_bond.apply_impact(direct_impact)
        impacts[traded_bond.id] = direct_impact
        
        # Cross-impact on related bonds
        for bond in all_bonds:
            if bond.id == traded_bond.id:
                continue
            
            # Impact same issuer bonds
            if bond.issuer == traded_bond.issuer:
                impact = direct_impact * self.impact_cross_issuer
                bond.apply_impact(impact)
                impacts[bond.id] = impact
            
            # Impact same sector bonds
            elif bond.sector == traded_bond.sector:
                impact = direct_impact * self.impact_cross_sector
                bond.apply_impact(impact)
                impacts[bond.id] = impact
        
        return impacts
