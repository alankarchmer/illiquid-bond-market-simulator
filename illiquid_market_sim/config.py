"""
Configuration settings for the illiquid bond market simulator.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    
    # Market
    num_bonds: int = 50
    num_steps: int = 100
    random_seed: int = 42
    
    # Market dynamics
    market_volatility: float = 0.02  # daily volatility
    jump_probability: float = 0.05  # probability of credit event per step
    sector_correlation: float = 0.3  # correlation within sectors
    
    # Market impact
    base_impact_coeff: float = 0.001  # base impact per unit size
    impact_decay: float = 0.5  # how fast impact decays
    cross_impact_factor: float = 0.3  # impact on related bonds
    
    # Dealer
    base_spread_bps: float = 50  # base bid-ask spread in bps
    illiquidity_spread_multiplier: float = 2.0
    inventory_risk_penalty: float = 0.01  # penalty per unit inventory
    max_inventory_per_bond: float = 10.0
    
    # Clients
    num_real_money_clients: int = 3
    num_hedge_fund_clients: int = 2
    num_fisher_clients: int = 2
    num_noise_clients: int = 3
    
    # RFQ generation
    rfq_prob_per_client: float = 0.1  # probability each client sends RFQ per step
    mean_trade_size: float = 1.0
    trade_size_std: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'num_bonds': self.num_bonds,
            'num_steps': self.num_steps,
            'random_seed': self.random_seed,
            'market_volatility': self.market_volatility,
            'jump_probability': self.jump_probability,
            'base_impact_coeff': self.base_impact_coeff,
            'base_spread_bps': self.base_spread_bps,
        }


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig()
