"""
Metrics and evaluation for simulation results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from illiquid_market_sim.rfq import Trade


@dataclass
class SimulationResult:
    """
    Container for simulation results and metrics.
    
    Attributes:
        total_steps: Number of timesteps simulated
        total_rfqs: Total RFQs received
        total_trades: Total trades executed
        fill_ratio: Overall fill ratio
        final_pnl: Final total P&L
        realized_pnl: Realized P&L from closed positions
        unrealized_pnl: Unrealized P&L from open positions
        impact_cost: Estimated cost from market impact
        inventory_risk: Final inventory risk metric
        trades: List of all trades
        rfq_history: List of all RFQs
        pnl_history: Time series of P&L
        client_breakdown: Per-client statistics
        position_summary: Summary of final positions
    """
    
    total_steps: int
    total_rfqs: int
    total_trades: int
    fill_ratio: float
    
    final_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    impact_cost: float
    inventory_risk: float
    
    trades: List[Trade] = field(default_factory=list)
    rfq_history: List[Dict] = field(default_factory=list)
    pnl_history: List[Dict] = field(default_factory=list)
    client_breakdown: Dict[str, Dict] = field(default_factory=dict)
    position_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'total_steps': self.total_steps,
            'total_rfqs': self.total_rfqs,
            'total_trades': self.total_trades,
            'fill_ratio': self.fill_ratio,
            'final_pnl': self.final_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'impact_cost': self.impact_cost,
            'inventory_risk': self.inventory_risk,
        }


def calculate_impact_cost(trades: List[Trade], bonds_dict: Dict) -> float:
    """
    Estimate the total cost from market impact.
    
    This is a rough estimate of how much we "hurt ourselves" by moving the market.
    """
    # Simple estimation: sum of (trade size * volatility)
    # In reality, this would be more sophisticated
    total_impact = 0.0
    
    for trade in trades:
        bond = bonds_dict.get(trade.bond_id)
        if bond:
            # Impact proportional to size and inversely to liquidity
            liquidity = max(bond.liquidity, 0.01)
            impact = abs(trade.size) * bond.volatility / liquidity
            total_impact += impact
    
    return total_impact


def summarize_simulation(result: SimulationResult) -> str:
    """
    Create a human-readable summary of simulation results.
    
    Args:
        result: SimulationResult object
        
    Returns:
        Multi-line summary string
    """
    summary = []
    summary.append("=" * 70)
    summary.append("SIMULATION SUMMARY")
    summary.append("=" * 70)
    summary.append("")
    
    # Overview
    summary.append(f"Total Steps:       {result.total_steps}")
    summary.append(f"Total RFQs:        {result.total_rfqs}")
    summary.append(f"Total Trades:      {result.total_trades}")
    summary.append(f"Fill Ratio:        {result.fill_ratio:.1%}")
    summary.append("")
    
    # P&L
    summary.append("P&L BREAKDOWN")
    summary.append("-" * 70)
    summary.append(f"Final Total P&L:   {result.final_pnl:+.2f}")
    summary.append(f"  Realized:        {result.realized_pnl:+.2f}")
    summary.append(f"  Unrealized:      {result.unrealized_pnl:+.2f}")
    summary.append(f"Impact Cost:       {result.impact_cost:.2f}")
    summary.append(f"Inventory Risk:    {result.inventory_risk:.2f}")
    summary.append("")
    
    # Per-client breakdown
    if result.client_breakdown:
        summary.append("CLIENT BREAKDOWN")
        summary.append("-" * 70)
        summary.append(f"{'Client':<15} {'Type':<12} {'RFQs':<8} {'Trades':<8} {'Fill%':<8} {'Avg Edge':<10}")
        summary.append("-" * 70)
        
        for client_id, stats in sorted(result.client_breakdown.items()):
            summary.append(
                f"{client_id:<15} "
                f"{stats.get('type', 'N/A'):<12} "
                f"{stats.get('rfq_count', 0):<8} "
                f"{stats.get('trade_count', 0):<8} "
                f"{stats.get('fill_ratio', 0):<8.1%} "
                f"{stats.get('avg_edge', 0):+<10.2f}"
            )
        summary.append("")
    
    # Position summary
    if result.position_summary:
        summary.append("POSITION SUMMARY")
        summary.append("-" * 70)
        summary.append(f"Active Positions:  {result.position_summary.get('num_positions', 0)}")
        summary.append(f"Total Trades:      {result.position_summary.get('total_trades', 0)}")
        summary.append(f"Inventory Risk:    {result.position_summary.get('inventory_risk', 0):.2f}")
        summary.append("")
    
    # P&L over time (show key points)
    if result.pnl_history:
        summary.append("P&L TRAJECTORY (sampled)")
        summary.append("-" * 70)
        
        # Sample every N steps to show progression
        n = len(result.pnl_history)
        sample_indices = [0, n//4, n//2, 3*n//4, n-1]
        
        for idx in sample_indices:
            if idx < len(result.pnl_history):
                snapshot = result.pnl_history[idx]
                step = idx
                total_pnl = snapshot.get('total', 0)
                summary.append(f"  Step {step:<6}: Total P&L = {total_pnl:+.2f}")
        summary.append("")
    
    # Trade examples
    if result.trades:
        summary.append(f"SAMPLE TRADES (showing first 5 of {len(result.trades)})")
        summary.append("-" * 70)
        for i, trade in enumerate(result.trades[:5]):
            summary.append(f"  {trade}")
        summary.append("")
    
    summary.append("=" * 70)
    
    return "\n".join(summary)


def analyze_client_toxicity(result: SimulationResult) -> Dict[str, List[str]]:
    """
    Analyze which clients appear toxic (low fill ratio, high edge capture).
    
    Returns:
        Dict with 'toxic', 'neutral', 'good' lists of client IDs
    """
    toxic = []
    neutral = []
    good = []
    
    for client_id, stats in result.client_breakdown.items():
        fill_ratio = stats.get('fill_ratio', 0)
        avg_edge = stats.get('avg_edge', 0)
        rfq_count = stats.get('rfq_count', 0)
        
        if rfq_count < 5:
            neutral.append(client_id)
            continue
        
        # Toxic: low fill ratio + high edge
        if fill_ratio < 0.3 and avg_edge > 0.5:
            toxic.append(client_id)
        # Good: high fill ratio + low/negative edge
        elif fill_ratio > 0.5 and avg_edge < 0.2:
            good.append(client_id)
        else:
            neutral.append(client_id)
    
    return {
        'toxic': toxic,
        'neutral': neutral,
        'good': good
    }


def calculate_sharpe_ratio(pnl_history: List[Dict], risk_free_rate: float = 0.0) -> float:
    """
    Calculate a simple Sharpe ratio from P&L history.
    
    Args:
        pnl_history: List of P&L snapshots
        risk_free_rate: Risk-free rate (daily)
        
    Returns:
        Sharpe ratio
    """
    if len(pnl_history) < 2:
        return 0.0
    
    # Calculate returns
    returns = []
    for i in range(1, len(pnl_history)):
        prev_pnl = pnl_history[i-1].get('total', 0)
        curr_pnl = pnl_history[i].get('total', 0)
        ret = curr_pnl - prev_pnl
        returns.append(ret)
    
    if not returns:
        return 0.0
    
    # Mean and std
    mean_return = sum(returns) / len(returns)
    if len(returns) < 2:
        return 0.0
    
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_return = variance ** 0.5
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe


def calculate_max_drawdown(pnl_history: List[Dict]) -> float:
    """
    Calculate maximum drawdown from P&L history.
    
    Returns:
        Maximum drawdown (positive number representing loss from peak)
    """
    if not pnl_history:
        return 0.0
    
    pnls = [snapshot.get('total', 0) for snapshot in pnl_history]
    
    peak = pnls[0]
    max_dd = 0.0
    
    for pnl in pnls:
        if pnl > peak:
            peak = pnl
        dd = peak - pnl
        if dd > max_dd:
            max_dd = dd
    
    return max_dd
