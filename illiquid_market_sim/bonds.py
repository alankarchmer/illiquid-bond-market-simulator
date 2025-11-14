"""
Bond model for the illiquid market simulator.
"""

import random
from dataclasses import dataclass, field
from typing import Literal, Dict, Optional


SECTORS = ["IG", "HY", "EM"]
RATINGS = {
    "IG": ["AAA", "AA", "A", "BBB"],
    "HY": ["BB", "B", "CCC"],
    "EM": ["BB", "B", "CCC", "CC"]
}
ISSUERS = [
    "Acme Corp", "TechGiant Inc", "RetailCo", "EnergyPro", "BankHoldings",
    "Pharma Solutions", "AutoMakers", "Telecom Group", "Construction Ltd",
    "Food Industries", "MediaCorp", "Mining Co", "RealEstate Dev",
    "Transport Systems", "Utilities Power", "Chemicals Inc", "Defense Tech",
    "Hospitality Group", "Insurance Co", "Manufacturing Inc"
]


@dataclass
class Bond:
    """
    Represents an illiquid corporate bond.
    
    Attributes:
        id: Unique bond identifier
        issuer: Name of the issuing company
        sector: Sector classification (IG, HY, EM)
        rating: Credit rating
        maturity_years: Years to maturity
        liquidity: Liquidity measure (0=dead, 1=most liquid)
        volatility: Daily price volatility (standard deviation)
        base_spread: Base spread over risk-free in bps
    """
    
    id: str
    issuer: str
    sector: Literal["IG", "HY", "EM"]
    rating: str
    maturity_years: float
    liquidity: float  # 0 to 1, where 0 is illiquid
    volatility: float
    base_spread: float  # in basis points
    
    # Internal state (not directly observable by dealer)
    _true_fair_price: float = field(default=100.0, repr=False)
    _last_traded_price: Optional[float] = field(default=None, repr=False)
    _accumulated_impact: float = field(default=0.0, repr=False)
    
    def get_true_fair_price(self) -> float:
        """
        Get the hidden true fair price of the bond.
        This includes market factors and accumulated impact.
        """
        return self._true_fair_price + self._accumulated_impact
    
    def update_fair_price(self, new_price: float) -> None:
        """Update the true fair price based on market factors."""
        self._true_fair_price = new_price
    
    def apply_impact(self, impact: float) -> None:
        """Apply market impact to this bond's fair price."""
        self._accumulated_impact += impact
    
    def record_trade(self, price: float) -> None:
        """Record a trade at the given price."""
        self._last_traded_price = price
    
    def get_last_traded_price(self) -> Optional[float]:
        """Get the last traded price (observable signal)."""
        return self._last_traded_price
    
    def get_naive_mid(self, market_factors: Dict[str, float]) -> float:
        """
        Compute a naive mid price estimate based on observable factors.
        This is what the dealer uses (imperfect estimate of fair value).
        """
        # Start with last traded price if available
        if self._last_traded_price is not None:
            base = self._last_traded_price
        else:
            base = 100.0
        
        # Adjust by sector factor
        sector_adjustment = market_factors.get(f"level_{self.sector}", 0.0)
        
        # Add some noise to make it imperfect
        noise = random.gauss(0, self.volatility * 0.5)
        
        return base + sector_adjustment + noise
    
    def __hash__(self) -> int:
        return hash(self.id)


def generate_bond_universe(n_bonds: int, seed: Optional[int] = None) -> list[Bond]:
    """
    Generate a synthetic universe of illiquid bonds.
    
    Args:
        n_bonds: Number of bonds to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of Bond objects
    """
    if seed is not None:
        random.seed(seed)
    
    bonds = []
    
    for i in range(n_bonds):
        # Choose sector
        sector = random.choice(SECTORS)
        
        # Choose rating based on sector
        rating = random.choice(RATINGS[sector])
        
        # Choose issuer
        issuer = random.choice(ISSUERS)
        
        # Generate characteristics
        maturity_years = random.uniform(1, 30)
        
        # Liquidity: IG more liquid than HY/EM
        if sector == "IG":
            liquidity = random.uniform(0.3, 0.8)
        elif sector == "HY":
            liquidity = random.uniform(0.1, 0.5)
        else:  # EM
            liquidity = random.uniform(0.05, 0.4)
        
        # Volatility: depends on sector and rating
        base_vol = {
            "IG": 0.005,
            "HY": 0.015,
            "EM": 0.020
        }[sector]
        
        # Worse ratings = higher volatility
        rating_multiplier = {
            "AAA": 0.5, "AA": 0.7, "A": 0.9, "BBB": 1.0,
            "BB": 1.2, "B": 1.5, "CCC": 2.0, "CC": 2.5
        }.get(rating, 1.0)
        
        volatility = base_vol * rating_multiplier
        
        # Base spread: depends on rating and maturity
        rating_spread = {
            "AAA": 50, "AA": 75, "A": 100, "BBB": 150,
            "BB": 300, "B": 500, "CCC": 800, "CC": 1200
        }.get(rating, 500)
        
        maturity_premium = maturity_years * 5  # 5bps per year
        base_spread = rating_spread + maturity_premium
        
        # Initial fair price with some randomness
        initial_price = 100 + random.gauss(0, volatility * 10)
        
        bond = Bond(
            id=f"BOND{i:03d}",
            issuer=f"{issuer} {i//3}",  # Reuse issuers to create correlation
            sector=sector,
            rating=rating,
            maturity_years=round(maturity_years, 1),
            liquidity=round(liquidity, 2),
            volatility=round(volatility, 4),
            base_spread=round(base_spread, 1),
            _true_fair_price=round(initial_price, 2)
        )
        
        bonds.append(bond)
    
    return bonds


def get_bonds_by_issuer(bonds: list[Bond], issuer: str) -> list[Bond]:
    """Get all bonds from a specific issuer."""
    return [b for b in bonds if b.issuer == issuer]


def get_bonds_by_sector(bonds: list[Bond], sector: str) -> list[Bond]:
    """Get all bonds in a specific sector."""
    return [b for b in bonds if b.sector == sector]
