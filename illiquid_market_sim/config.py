"""
Configuration settings for the illiquid bond market simulator.

This module provides configuration dataclasses for:
- Simulation parameters (market, clients, dynamics)
- Market regime presets (normal, stressed, crisis)
- Bond universe characteristics
- RL-specific settings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import json
import os


class MarketRegime(str, Enum):
    """Market regime presets for different simulation scenarios."""
    NORMAL = "normal"
    STRESSED = "stressed"
    CRISIS = "crisis"
    LIQUID = "liquid"
    ILLIQUID = "illiquid"


@dataclass
class BondUniverseConfig:
    """
    Configuration for bond universe generation.
    
    Controls the distribution of bond characteristics.
    """
    # Sector distribution (weights, will be normalized)
    sector_weights: Dict[str, float] = field(default_factory=lambda: {
        "IG": 0.5,
        "HY": 0.3,
        "EM": 0.2
    })
    
    # Maturity distribution
    min_maturity_years: float = 1.0
    max_maturity_years: float = 30.0
    
    # Liquidity ranges by sector
    liquidity_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "IG": (0.3, 0.8),
        "HY": (0.1, 0.5),
        "EM": (0.05, 0.4)
    })
    
    # Volatility base by sector
    volatility_base: Dict[str, float] = field(default_factory=lambda: {
        "IG": 0.005,
        "HY": 0.015,
        "EM": 0.020
    })


@dataclass
class ClientConfig:
    """Configuration for client behavior."""
    
    # Client counts
    num_real_money_clients: int = 3
    num_hedge_fund_clients: int = 2
    num_fisher_clients: int = 2
    num_noise_clients: int = 3
    
    # RFQ generation
    rfq_prob_per_client: float = 0.1
    mean_trade_size: float = 1.0
    trade_size_std: float = 0.5
    
    # Client behavior parameters
    hedge_fund_information_quality: float = 0.8
    fisher_fishing_probability: float = 0.80
    noise_trade_probability: float = 0.40
    
    @property
    def total_clients(self) -> int:
        """Total number of clients."""
        return (
            self.num_real_money_clients +
            self.num_hedge_fund_clients +
            self.num_fisher_clients +
            self.num_noise_clients
        )


@dataclass
class MarketDynamicsConfig:
    """Configuration for market dynamics and impact."""
    
    # Volatility
    market_volatility: float = 0.02
    jump_probability: float = 0.05
    sector_correlation: float = 0.3
    
    # Market impact
    base_impact_coeff: float = 0.001
    impact_decay: float = 0.5
    cross_impact_factor: float = 0.3
    
    # Jump event parameters
    sector_shock_min: float = -5.0
    sector_shock_max: float = -2.0
    issuer_shock_min: float = -10.0
    issuer_shock_max: float = -3.0


@dataclass
class DealerConfig:
    """Configuration for dealer quoting behavior."""
    
    base_spread_bps: float = 50.0
    illiquidity_spread_multiplier: float = 2.0
    inventory_risk_penalty: float = 0.01
    max_inventory_per_bond: float = 10.0
    toxic_client_spread_multiplier: float = 1.5


@dataclass
class SimulationConfig:
    """
    Master configuration for simulation parameters.
    
    This is the main configuration class that combines all sub-configurations.
    It can be created directly, loaded from a preset, or loaded from a file.
    """
    
    # Core settings
    num_bonds: int = 50
    num_steps: int = 100
    random_seed: int = 42
    
    # Market dynamics
    market_volatility: float = 0.02
    jump_probability: float = 0.05
    sector_correlation: float = 0.3
    
    # Market impact
    base_impact_coeff: float = 0.001
    impact_decay: float = 0.5
    cross_impact_factor: float = 0.3
    
    # Dealer
    base_spread_bps: float = 50.0
    illiquidity_spread_multiplier: float = 2.0
    inventory_risk_penalty: float = 0.01
    max_inventory_per_bond: float = 10.0
    
    # Clients
    num_real_money_clients: int = 3
    num_hedge_fund_clients: int = 2
    num_fisher_clients: int = 2
    num_noise_clients: int = 3
    
    # RFQ generation
    rfq_prob_per_client: float = 0.1
    mean_trade_size: float = 1.0
    trade_size_std: float = 0.5
    
    # Sub-configurations (optional, for advanced use)
    bond_universe: Optional[BondUniverseConfig] = None
    client_config: Optional[ClientConfig] = None
    market_dynamics: Optional[MarketDynamicsConfig] = None
    dealer_config: Optional[DealerConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'num_bonds': self.num_bonds,
            'num_steps': self.num_steps,
            'random_seed': self.random_seed,
            'market_volatility': self.market_volatility,
            'jump_probability': self.jump_probability,
            'sector_correlation': self.sector_correlation,
            'base_impact_coeff': self.base_impact_coeff,
            'impact_decay': self.impact_decay,
            'cross_impact_factor': self.cross_impact_factor,
            'base_spread_bps': self.base_spread_bps,
            'illiquidity_spread_multiplier': self.illiquidity_spread_multiplier,
            'inventory_risk_penalty': self.inventory_risk_penalty,
            'max_inventory_per_bond': self.max_inventory_per_bond,
            'num_real_money_clients': self.num_real_money_clients,
            'num_hedge_fund_clients': self.num_hedge_fund_clients,
            'num_fisher_clients': self.num_fisher_clients,
            'num_noise_clients': self.num_noise_clients,
            'rfq_prob_per_client': self.rfq_prob_per_client,
            'mean_trade_size': self.mean_trade_size,
            'trade_size_std': self.trade_size_std,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create config from dictionary."""
        # Filter to only valid fields
        valid_fields = {
            'num_bonds', 'num_steps', 'random_seed',
            'market_volatility', 'jump_probability', 'sector_correlation',
            'base_impact_coeff', 'impact_decay', 'cross_impact_factor',
            'base_spread_bps', 'illiquidity_spread_multiplier',
            'inventory_risk_penalty', 'max_inventory_per_bond',
            'num_real_money_clients', 'num_hedge_fund_clients',
            'num_fisher_clients', 'num_noise_clients',
            'rfq_prob_per_client', 'mean_trade_size', 'trade_size_std',
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SimulationConfig":
        """Create config from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> "SimulationConfig":
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())


# -----------------------------------------------------------------------------
# Preset Configurations
# -----------------------------------------------------------------------------

def get_regime_config(regime: MarketRegime) -> SimulationConfig:
    """
    Get a preset configuration for a specific market regime.
    
    Args:
        regime: The market regime to configure for
    
    Returns:
        SimulationConfig tuned for the specified regime
    """
    if regime == MarketRegime.NORMAL:
        return SimulationConfig(
            market_volatility=0.02,
            jump_probability=0.05,
            base_impact_coeff=0.001,
            base_spread_bps=50.0,
        )
    
    elif regime == MarketRegime.STRESSED:
        return SimulationConfig(
            market_volatility=0.05,
            jump_probability=0.15,
            base_impact_coeff=0.003,
            base_spread_bps=100.0,
            illiquidity_spread_multiplier=3.0,
        )
    
    elif regime == MarketRegime.CRISIS:
        return SimulationConfig(
            market_volatility=0.10,
            jump_probability=0.30,
            base_impact_coeff=0.010,
            base_spread_bps=200.0,
            illiquidity_spread_multiplier=5.0,
            inventory_risk_penalty=0.05,
        )
    
    elif regime == MarketRegime.LIQUID:
        return SimulationConfig(
            market_volatility=0.01,
            jump_probability=0.02,
            base_impact_coeff=0.0005,
            base_spread_bps=25.0,
            illiquidity_spread_multiplier=1.0,
        )
    
    elif regime == MarketRegime.ILLIQUID:
        return SimulationConfig(
            market_volatility=0.03,
            jump_probability=0.08,
            base_impact_coeff=0.005,
            base_spread_bps=100.0,
            illiquidity_spread_multiplier=4.0,
            rfq_prob_per_client=0.05,
        )
    
    return SimulationConfig()


# Preset configurations for common scenarios
PRESET_CONFIGS: Dict[str, SimulationConfig] = {
    "default": SimulationConfig(),
    "normal": get_regime_config(MarketRegime.NORMAL),
    "stressed": get_regime_config(MarketRegime.STRESSED),
    "crisis": get_regime_config(MarketRegime.CRISIS),
    "liquid": get_regime_config(MarketRegime.LIQUID),
    "illiquid": get_regime_config(MarketRegime.ILLIQUID),
    
    # RL training presets
    "rl_easy": SimulationConfig(
        num_bonds=20,
        num_steps=50,
        market_volatility=0.01,
        jump_probability=0.02,
        num_real_money_clients=2,
        num_hedge_fund_clients=1,
        num_fisher_clients=1,
        num_noise_clients=2,
    ),
    "rl_medium": SimulationConfig(
        num_bonds=50,
        num_steps=100,
        market_volatility=0.02,
        jump_probability=0.05,
    ),
    "rl_hard": SimulationConfig(
        num_bonds=100,
        num_steps=200,
        market_volatility=0.05,
        jump_probability=0.10,
        num_hedge_fund_clients=4,
        num_fisher_clients=4,
    ),
}


def get_preset(name: str) -> SimulationConfig:
    """
    Get a preset configuration by name.
    
    Args:
        name: Preset name (e.g., "default", "stressed", "rl_easy")
    
    Returns:
        SimulationConfig for the preset
    
    Raises:
        KeyError: If preset name is not found
    """
    if name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESET_CONFIGS[name]


def list_presets() -> List[str]:
    """List available preset names."""
    return list(PRESET_CONFIGS.keys())


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig()
