"""
Scenario system for the illiquid bond market simulator.

Each scenario represents a specific market regime with its own parameters
for volatility, drift, flow patterns, client mix, and shock probabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable


@dataclass
class ScenarioConfig:
    """
    Configuration for a market scenario.
    
    A scenario defines the market regime, flow environment, and dynamics
    that drive the simulation. Each scenario acts like a "level" with
    its own characteristics.
    """
    
    # Identification
    name: str
    description: str
    
    # Market dynamics
    base_volatility: float = 0.02  # Baseline market volatility
    spread_drift_bps_per_step: float = 0.0  # Persistent drift (+ = wider, - = tighter)
    sector_shock_prob: float = 0.05  # Probability of sector-specific shock
    market_shock_prob: float = 0.02  # Probability of broad market shock
    idiosyncratic_event_prob: float = 0.03  # Probability of issuer-specific events
    liquidity_multiplier: float = 1.0  # >1 = less liquid (bigger impact per trade)
    
    # RFQ / Flow
    avg_rfq_per_step: int = 1  # Expected number of RFQs per step
    rfq_size_multiplier: float = 1.0  # Scale RFQ sizes up/down
    buy_sell_skew: float = 0.0  # -1.0 (all sells) to +1.0 (all buys), 0 = neutral
    
    # Client mix (fractions should roughly sum to 1.0)
    fraction_real_money: float = 0.30
    fraction_hedge_fund: float = 0.20
    fraction_fisher: float = 0.20
    fraction_noise: float = 0.30
    fraction_etf_arb: float = 0.0  # Reserved for future ETF/arb client type
    
    # Impact
    impact_coefficient: float = 0.001  # Base coefficient for price impact
    impact_cross_issuer: float = 0.3  # 0-1: how much impact spills to same issuer
    impact_cross_sector: float = 0.1  # 0-1: how much impact spills to same sector
    
    # Initial portfolio state (optional)
    initial_positions: Optional[Dict[str, float]] = None  # bond_id -> quantity
    
    # Time-varying parameters (for regime shift scenarios)
    regime_shift_callback: Optional[Callable[[int, int], 'ScenarioConfig']] = field(
        default=None, repr=False
    )
    
    def get_total_client_count(self, base_count: int = 10) -> Dict[str, int]:
        """
        Convert client fractions to actual counts.
        
        Args:
            base_count: Total number of clients to create
            
        Returns:
            Dictionary with counts per client type
        """
        total_fraction = (
            self.fraction_real_money + 
            self.fraction_hedge_fund + 
            self.fraction_fisher + 
            self.fraction_noise
        )
        
        if total_fraction > 0:
            return {
                'real_money': max(1, round(self.fraction_real_money / total_fraction * base_count)),
                'hedge_fund': max(1, round(self.fraction_hedge_fund / total_fraction * base_count)),
                'fisher': max(0, round(self.fraction_fisher / total_fraction * base_count)),
                'noise': max(0, round(self.fraction_noise / total_fraction * base_count))
            }
        else:
            # Default if fractions are all zero
            return {'real_money': 3, 'hedge_fund': 2, 'fisher': 2, 'noise': 3}


def get_scenarios() -> Dict[str, ScenarioConfig]:
    """
    Get all available scenarios.
    
    Returns:
        Dictionary mapping scenario name to ScenarioConfig
    """
    scenarios = {}
    
    # Scenario 1: Quiet, well-behaved market
    scenarios['quiet_market'] = ScenarioConfig(
        name='quiet_market',
        description='Stable spreads, low volatility, moderate flow. Good for calibration.',
        base_volatility=0.015,
        spread_drift_bps_per_step=0.0,
        sector_shock_prob=0.02,
        market_shock_prob=0.01,
        idiosyncratic_event_prob=0.02,
        liquidity_multiplier=1.0,
        avg_rfq_per_step=1,
        rfq_size_multiplier=1.0,
        buy_sell_skew=0.0,
        fraction_real_money=0.30,
        fraction_hedge_fund=0.20,
        fraction_fisher=0.20,
        fraction_noise=0.30,
        impact_coefficient=0.0008,
        impact_cross_issuer=0.25,
        impact_cross_sector=0.08
    )
    
    # Scenario 2: Slow grind tighter (risk-on rally)
    scenarios['grind_tighter'] = ScenarioConfig(
        name='grind_tighter',
        description='Spreads slowly tightening, more buy RFQs. Hedge funds lead the rally.',
        base_volatility=0.018,
        spread_drift_bps_per_step=-0.15,  # Negative = tighter spreads
        sector_shock_prob=0.01,
        market_shock_prob=0.005,
        idiosyncratic_event_prob=0.01,
        liquidity_multiplier=0.9,  # Slightly better liquidity
        avg_rfq_per_step=1,
        rfq_size_multiplier=1.1,
        buy_sell_skew=0.4,  # Bias toward buys
        fraction_real_money=0.25,
        fraction_hedge_fund=0.35,  # More hedge funds
        fraction_fisher=0.15,
        fraction_noise=0.25,
        impact_coefficient=0.001,
        impact_cross_issuer=0.3,
        impact_cross_sector=0.1
    )
    
    # Scenario 3: Slow grind wider (risk-off drift)
    scenarios['grind_wider'] = ScenarioConfig(
        name='grind_wider',
        description='Spreads gradually widening, more sells. Real-money offloading risk.',
        base_volatility=0.022,
        spread_drift_bps_per_step=0.18,  # Positive = wider spreads
        sector_shock_prob=0.03,
        market_shock_prob=0.015,
        idiosyncratic_event_prob=0.02,
        liquidity_multiplier=1.2,  # Deteriorating liquidity
        avg_rfq_per_step=1,
        rfq_size_multiplier=1.15,
        buy_sell_skew=-0.35,  # Bias toward sells
        fraction_real_money=0.40,  # More real-money selling
        fraction_hedge_fund=0.25,
        fraction_fisher=0.15,
        fraction_noise=0.20,
        impact_coefficient=0.0012,
        impact_cross_issuer=0.35,
        impact_cross_sector=0.12
    )
    
    # Scenario 4: Sudden credit shock / mini-crisis
    scenarios['credit_shock'] = ScenarioConfig(
        name='credit_shock',
        description='Volatility spikes, spreads gap wider. Panic selling, liquidity dries up.',
        base_volatility=0.045,  # High volatility
        spread_drift_bps_per_step=0.5,  # Strong widening
        sector_shock_prob=0.08,
        market_shock_prob=0.10,  # Frequent broad shocks
        idiosyncratic_event_prob=0.05,
        liquidity_multiplier=2.0,  # Much worse liquidity
        avg_rfq_per_step=2,  # Panic flow
        rfq_size_multiplier=1.3,
        buy_sell_skew=-0.6,  # Strong sell bias
        fraction_real_money=0.35,
        fraction_hedge_fund=0.30,
        fraction_fisher=0.10,
        fraction_noise=0.25,
        impact_coefficient=0.003,  # High impact
        impact_cross_issuer=0.5,
        impact_cross_sector=0.25
    )
    
    # Scenario 5: Sector-specific blow-up
    scenarios['sector_blowup'] = ScenarioConfig(
        name='sector_blowup',
        description='One sector hit hard, others stable. Strong cross-issuer impact in stressed sector.',
        base_volatility=0.025,
        spread_drift_bps_per_step=0.1,
        sector_shock_prob=0.15,  # Frequent sector shocks
        market_shock_prob=0.02,
        idiosyncratic_event_prob=0.03,
        liquidity_multiplier=1.4,
        avg_rfq_per_step=1,
        rfq_size_multiplier=1.2,
        buy_sell_skew=-0.3,
        fraction_real_money=0.35,
        fraction_hedge_fund=0.30,
        fraction_fisher=0.15,
        fraction_noise=0.20,
        impact_coefficient=0.0015,
        impact_cross_issuer=0.6,  # High cross-issuer impact
        impact_cross_sector=0.15
    )
    
    # Scenario 6: Idiosyncratic issuer event
    scenarios['issuer_event'] = ScenarioConfig(
        name='issuer_event',
        description='Issuer-specific rumors/downgrades. Jumpy, nonlinear fair values.',
        base_volatility=0.020,
        spread_drift_bps_per_step=0.05,
        sector_shock_prob=0.02,
        market_shock_prob=0.01,
        idiosyncratic_event_prob=0.12,  # High issuer event probability
        liquidity_multiplier=1.3,
        avg_rfq_per_step=1,
        rfq_size_multiplier=1.0,
        buy_sell_skew=-0.15,
        fraction_real_money=0.25,
        fraction_hedge_fund=0.40,  # Hedge funds trade around events
        fraction_fisher=0.15,
        fraction_noise=0.20,
        impact_coefficient=0.0012,
        impact_cross_issuer=0.7,  # Very high for same issuer
        impact_cross_sector=0.08
    )
    
    # Scenario 7: Fisher RFQ onslaught
    scenarios['fisher_onslaught'] = ScenarioConfig(
        name='fisher_onslaught',
        description='High volume of fishing RFQs extracting pricing information.',
        base_volatility=0.018,
        spread_drift_bps_per_step=0.0,
        sector_shock_prob=0.02,
        market_shock_prob=0.01,
        idiosyncratic_event_prob=0.02,
        liquidity_multiplier=1.0,
        avg_rfq_per_step=2,  # High RFQ volume
        rfq_size_multiplier=0.9,
        buy_sell_skew=0.0,
        fraction_real_money=0.15,
        fraction_hedge_fund=0.15,
        fraction_fisher=0.50,  # Mostly fishers
        fraction_noise=0.20,
        impact_coefficient=0.0008,
        impact_cross_issuer=0.25,
        impact_cross_sector=0.08
    )
    
    # Scenario 8: Informed hedge fund flow
    scenarios['informed_flow'] = ScenarioConfig(
        name='informed_flow',
        description='Toxic, informed hedge fund clients. RFQs anticipate spread moves.',
        base_volatility=0.028,
        spread_drift_bps_per_step=0.08,
        sector_shock_prob=0.04,
        market_shock_prob=0.03,
        idiosyncratic_event_prob=0.04,
        liquidity_multiplier=1.15,
        avg_rfq_per_step=1,
        rfq_size_multiplier=1.2,
        buy_sell_skew=0.0,
        fraction_real_money=0.15,
        fraction_hedge_fund=0.55,  # Dominated by hedge funds
        fraction_fisher=0.10,
        fraction_noise=0.20,
        impact_coefficient=0.0011,
        impact_cross_issuer=0.35,
        impact_cross_sector=0.12
    )
    
    # Scenario 9: Dominant real-money account
    scenarios['big_real_money'] = ScenarioConfig(
        name='big_real_money',
        description='One large real-money client with big, lumpy, price-insensitive trades.',
        base_volatility=0.020,
        spread_drift_bps_per_step=0.02,
        sector_shock_prob=0.025,
        market_shock_prob=0.015,
        idiosyncratic_event_prob=0.02,
        liquidity_multiplier=1.1,
        avg_rfq_per_step=1,
        rfq_size_multiplier=2.5,  # Large trade sizes
        buy_sell_skew=0.0,
        fraction_real_money=0.60,  # Dominated by real-money
        fraction_hedge_fund=0.15,
        fraction_fisher=0.10,
        fraction_noise=0.15,
        impact_coefficient=0.0014,  # Big trades move market
        impact_cross_issuer=0.4,
        impact_cross_sector=0.15
    )
    
    # Scenario 10: ETF/index arb / rebalancing wave
    scenarios['etf_rebalance'] = ScenarioConfig(
        name='etf_rebalance',
        description='ETF-like flow: many small RFQs across baskets. Low information content.',
        base_volatility=0.016,
        spread_drift_bps_per_step=0.0,
        sector_shock_prob=0.015,
        market_shock_prob=0.01,
        idiosyncratic_event_prob=0.015,
        liquidity_multiplier=0.95,
        avg_rfq_per_step=3,  # High RFQ volume
        rfq_size_multiplier=0.6,  # Small sizes
        buy_sell_skew=0.0,  # Balanced
        fraction_real_money=0.20,
        fraction_hedge_fund=0.10,
        fraction_fisher=0.10,
        fraction_noise=0.60,  # Treat ETF arb as noise
        impact_coefficient=0.0006,
        impact_cross_issuer=0.20,
        impact_cross_sector=0.10
    )
    
    # Scenario 11: Liquidity dry-up (holiday / off-hours)
    scenarios['liquidity_dryup'] = ScenarioConfig(
        name='liquidity_dryup',
        description='Low RFQ volume, thin liquidity. Each trade moves the market more.',
        base_volatility=0.018,
        spread_drift_bps_per_step=0.0,
        sector_shock_prob=0.02,
        market_shock_prob=0.01,
        idiosyncratic_event_prob=0.015,
        liquidity_multiplier=2.5,  # Very illiquid
        avg_rfq_per_step=0,  # Low volume (will be handled specially)
        rfq_size_multiplier=1.0,
        buy_sell_skew=0.0,
        fraction_real_money=0.35,
        fraction_hedge_fund=0.25,
        fraction_fisher=0.20,
        fraction_noise=0.20,
        impact_coefficient=0.0025,  # High impact per trade
        impact_cross_issuer=0.4,
        impact_cross_sector=0.15
    )
    
    # Scenario 12: Month-end / index rebalance week
    scenarios['month_end'] = ScenarioConfig(
        name='month_end',
        description='Distorted flow around benchmark names. Temporary dislocations.',
        base_volatility=0.022,
        spread_drift_bps_per_step=0.05,
        sector_shock_prob=0.03,
        market_shock_prob=0.02,
        idiosyncratic_event_prob=0.02,
        liquidity_multiplier=1.3,
        avg_rfq_per_step=2,
        rfq_size_multiplier=1.3,
        buy_sell_skew=0.15,
        fraction_real_money=0.35,
        fraction_hedge_fund=0.15,
        fraction_fisher=0.15,
        fraction_noise=0.35,  # Mix of arb and noise
        impact_coefficient=0.0013,
        impact_cross_issuer=0.35,
        impact_cross_sector=0.12
    )
    
    # Scenario 13: Start very short a sector
    scenarios['short_squeeze'] = ScenarioConfig(
        name='short_squeeze',
        description='Initial short position in a sector. Stable or tightening spreads force covering.',
        base_volatility=0.020,
        spread_drift_bps_per_step=-0.12,  # Tightening = painful for shorts
        sector_shock_prob=0.02,
        market_shock_prob=0.01,
        idiosyncratic_event_prob=0.025,
        liquidity_multiplier=1.15,
        avg_rfq_per_step=1,
        rfq_size_multiplier=1.0,
        buy_sell_skew=0.25,  # Some buy pressure
        fraction_real_money=0.30,
        fraction_hedge_fund=0.30,
        fraction_fisher=0.15,
        fraction_noise=0.25,
        impact_coefficient=0.0011,
        impact_cross_issuer=0.35,
        impact_cross_sector=0.12,
        # Note: initial_positions would be set dynamically in the simulator
        initial_positions={}  # Placeholder
    )
    
    # Scenario 14: Inventory overhang (long stuff nobody wants)
    scenarios['inventory_overhang'] = ScenarioConfig(
        name='inventory_overhang',
        description='Long illiquid bonds. Mild risk-off, clients selling/fishing.',
        base_volatility=0.024,
        spread_drift_bps_per_step=0.20,  # Widening = painful for longs
        sector_shock_prob=0.04,
        market_shock_prob=0.025,
        idiosyncratic_event_prob=0.03,
        liquidity_multiplier=1.6,  # Poor liquidity
        avg_rfq_per_step=1,
        rfq_size_multiplier=1.1,
        buy_sell_skew=-0.35,  # More sells
        fraction_real_money=0.35,
        fraction_hedge_fund=0.25,
        fraction_fisher=0.25,  # More fishers
        fraction_noise=0.15,
        impact_coefficient=0.0016,
        impact_cross_issuer=0.4,
        impact_cross_sector=0.15,
        initial_positions={}  # Placeholder
    )
    
    # Scenario 15: Regime shift mid-simulation
    def regime_shift_callback(current_step: int, total_steps: int) -> ScenarioConfig:
        """Switch from quiet to stressed halfway through."""
        if current_step < total_steps // 2:
            # First half: quiet
            return ScenarioConfig(
                name='regime_shift_quiet',
                description='Quiet phase (first half)',
                base_volatility=0.015,
                spread_drift_bps_per_step=0.0,
                sector_shock_prob=0.015,
                market_shock_prob=0.005,
                idiosyncratic_event_prob=0.01,
                liquidity_multiplier=0.9,
                avg_rfq_per_step=1,
                rfq_size_multiplier=0.9,
                buy_sell_skew=0.0,
                fraction_real_money=0.30,
                fraction_hedge_fund=0.20,
                fraction_fisher=0.20,
                fraction_noise=0.30,
                impact_coefficient=0.0007,
                impact_cross_issuer=0.25,
                impact_cross_sector=0.08
            )
        else:
            # Second half: stressed
            return ScenarioConfig(
                name='regime_shift_stressed',
                description='Stressed phase (second half)',
                base_volatility=0.040,
                spread_drift_bps_per_step=0.35,
                sector_shock_prob=0.08,
                market_shock_prob=0.08,
                idiosyncratic_event_prob=0.05,
                liquidity_multiplier=1.8,
                avg_rfq_per_step=2,
                rfq_size_multiplier=1.3,
                buy_sell_skew=-0.5,
                fraction_real_money=0.35,
                fraction_hedge_fund=0.30,
                fraction_fisher=0.10,
                fraction_noise=0.25,
                impact_coefficient=0.0025,
                impact_cross_issuer=0.5,
                impact_cross_sector=0.22
            )
    
    scenarios['regime_shift'] = ScenarioConfig(
        name='regime_shift',
        description='Regime shift: quiet first half, then volatility spike and stress.',
        base_volatility=0.015,  # Starting values (will shift)
        spread_drift_bps_per_step=0.0,
        sector_shock_prob=0.015,
        market_shock_prob=0.005,
        idiosyncratic_event_prob=0.01,
        liquidity_multiplier=0.9,
        avg_rfq_per_step=1,
        rfq_size_multiplier=0.9,
        buy_sell_skew=0.0,
        fraction_real_money=0.30,
        fraction_hedge_fund=0.20,
        fraction_fisher=0.20,
        fraction_noise=0.30,
        impact_coefficient=0.0007,
        impact_cross_issuer=0.25,
        impact_cross_sector=0.08,
        regime_shift_callback=regime_shift_callback
    )
    
    return scenarios


def list_scenarios() -> str:
    """
    Generate a formatted list of all available scenarios.
    
    Returns:
        String with formatted scenario information
    """
    scenarios = get_scenarios()
    
    output = []
    output.append("=" * 80)
    output.append("AVAILABLE SCENARIOS")
    output.append("=" * 80)
    output.append("")
    
    for i, (key, scenario) in enumerate(scenarios.items(), 1):
        output.append(f"{i:2d}. {scenario.name}")
        output.append(f"    {scenario.description}")
        output.append("")
    
    output.append("=" * 80)
    output.append(f"Total: {len(scenarios)} scenarios")
    output.append("")
    output.append("Usage: python cli.py --scenario SCENARIO_NAME")
    output.append("Example: python cli.py --scenario credit_shock --steps 200 --verbose")
    output.append("=" * 80)
    
    return "\n".join(output)
