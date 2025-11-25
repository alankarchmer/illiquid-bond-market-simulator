"""
Metrics and evaluation for simulation results.

This module provides comprehensive metrics for evaluating trading performance:
- P&L metrics (Sharpe, Sortino, max drawdown)
- Execution quality metrics (fill ratio, slippage)
- Risk metrics (inventory, VaR proxies)
- Client analysis (toxicity detection)
- RL-specific metrics (episode returns, learning curves)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import json
import math
from pathlib import Path

import numpy as np

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


# -----------------------------------------------------------------------------
# Enhanced RL Metrics
# -----------------------------------------------------------------------------

def calculate_sortino_ratio(
    pnl_history: List[Dict],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (penalizes only downside volatility).
    
    Args:
        pnl_history: List of P&L snapshots
        risk_free_rate: Risk-free rate
        target_return: Target return (usually 0)
    
    Returns:
        Sortino ratio
    """
    if len(pnl_history) < 2:
        return 0.0
    
    returns = []
    for i in range(1, len(pnl_history)):
        prev_pnl = pnl_history[i-1].get('total', 0)
        curr_pnl = pnl_history[i].get('total', 0)
        returns.append(curr_pnl - prev_pnl)
    
    if not returns:
        return 0.0
    
    mean_return = np.mean(returns)
    
    # Downside deviation (only negative returns)
    downside_returns = [r for r in returns if r < target_return]
    if not downside_returns:
        return float('inf') if mean_return > risk_free_rate else 0.0
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / downside_std


def calculate_calmar_ratio(
    pnl_history: List[Dict],
    annualization_factor: float = 252.0
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Args:
        pnl_history: List of P&L snapshots
        annualization_factor: Factor for annualizing returns
    
    Returns:
        Calmar ratio
    """
    if len(pnl_history) < 2:
        return 0.0
    
    # Calculate total return
    total_return = pnl_history[-1].get('total', 0) - pnl_history[0].get('total', 0)
    
    # Calculate max drawdown
    max_dd = calculate_max_drawdown(pnl_history)
    
    if max_dd == 0:
        return float('inf') if total_return > 0 else 0.0
    
    return total_return / max_dd


def calculate_win_rate(trades: List[Trade], bonds_dict: Dict) -> float:
    """
    Calculate win rate (fraction of profitable trades).
    
    Args:
        trades: List of trades
        bonds_dict: Dict mapping bond_id to Bond
    
    Returns:
        Win rate as fraction
    """
    if not trades:
        return 0.0
    
    profitable = 0
    for trade in trades:
        bond = bonds_dict.get(trade.bond_id)
        if bond:
            fair_value = bond.get_true_fair_price()
            if trade.side == "buy":
                # Bought - profitable if bought below fair
                if trade.price < fair_value:
                    profitable += 1
            else:
                # Sold - profitable if sold above fair
                if trade.price > fair_value:
                    profitable += 1
    
    return profitable / len(trades)


def calculate_average_edge(trades: List[Trade], bonds_dict: Dict) -> float:
    """
    Calculate average edge per trade.
    
    Args:
        trades: List of trades
        bonds_dict: Dict mapping bond_id to Bond
    
    Returns:
        Average edge (positive = profitable)
    """
    if not trades:
        return 0.0
    
    total_edge = 0.0
    for trade in trades:
        bond = bonds_dict.get(trade.bond_id)
        if bond:
            fair_value = bond.get_true_fair_price()
            if trade.side == "buy":
                edge = fair_value - trade.price
            else:
                edge = trade.price - fair_value
            total_edge += edge * trade.size
    
    return total_edge / len(trades)


def calculate_inventory_metrics(
    pnl_history: List[Dict],
    positions: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate inventory-related risk metrics.
    
    Args:
        pnl_history: P&L history
        positions: Position summary
    
    Returns:
        Dict of inventory metrics
    """
    metrics = {}
    
    # Current inventory
    if positions:
        total_inventory = sum(
            abs(p.get('quantity', 0)) 
            for p in positions.values() 
            if isinstance(p, dict)
        )
        metrics['total_inventory'] = total_inventory
        metrics['num_positions'] = len([
            p for p in positions.values() 
            if isinstance(p, dict) and abs(p.get('quantity', 0)) > 0.01
        ])
    
    return metrics


@dataclass
class RLMetrics:
    """
    Comprehensive metrics for RL training evaluation.
    
    Attributes:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        episode_pnls: List of final PnLs
        episode_trades: List of trade counts
        episode_fill_ratios: List of fill ratios
    """
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_pnls: List[float] = field(default_factory=list)
    episode_trades: List[int] = field(default_factory=list)
    episode_fill_ratios: List[float] = field(default_factory=list)
    
    def add_episode(
        self,
        reward: float,
        length: int,
        pnl: float = 0.0,
        trades: int = 0,
        fill_ratio: float = 0.0
    ) -> None:
        """Add episode results."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_pnls.append(pnl)
        self.episode_trades.append(trades)
        self.episode_fill_ratios.append(fill_ratio)
    
    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
        rewards = self.episode_rewards[-last_n:] if last_n else self.episode_rewards
        lengths = self.episode_lengths[-last_n:] if last_n else self.episode_lengths
        pnls = self.episode_pnls[-last_n:] if last_n else self.episode_pnls
        trades = self.episode_trades[-last_n:] if last_n else self.episode_trades
        fill_ratios = self.episode_fill_ratios[-last_n:] if last_n else self.episode_fill_ratios
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'mean_length': float(np.mean(lengths)),
            'mean_pnl': float(np.mean(pnls)) if pnls else 0.0,
            'mean_trades': float(np.mean(trades)) if trades else 0.0,
            'mean_fill_ratio': float(np.mean(fill_ratios)) if fill_ratios else 0.0,
            'n_episodes': len(rewards),
        }
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary for serialization."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'episode_fill_ratios': self.episode_fill_ratios,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, List]) -> 'RLMetrics':
        """Create from dictionary."""
        return cls(
            episode_rewards=data.get('episode_rewards', []),
            episode_lengths=data.get('episode_lengths', []),
            episode_pnls=data.get('episode_pnls', []),
            episode_trades=data.get('episode_trades', []),
            episode_fill_ratios=data.get('episode_fill_ratios', []),
        )
    
    def save(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'RLMetrics':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class MetricsLogger:
    """
    Logger for tracking metrics during training.
    
    Supports logging to console, file, and optional integration
    with experiment tracking tools.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_freq: int = 100,
        verbose: int = 1
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for saving logs
            log_freq: Frequency of logging (in steps)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_freq = log_freq
        self.verbose = verbose
        
        self.rl_metrics = RLMetrics()
        self.step_metrics: List[Dict[str, float]] = []
        self.current_step = 0
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_step(self, metrics: Dict[str, float]) -> None:
        """Log metrics for a single step."""
        self.current_step += 1
        metrics['step'] = self.current_step
        self.step_metrics.append(metrics)
        
        if self.verbose >= 2 and self.current_step % self.log_freq == 0:
            self._print_step_metrics(metrics)
    
    def log_episode(
        self,
        reward: float,
        length: int,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics for a completed episode."""
        pnl = info.get('total_pnl', 0.0) if info else 0.0
        trades = info.get('total_trades', 0) if info else 0
        fill_ratio = info.get('fill_ratio', 0.0) if info else 0.0
        
        self.rl_metrics.add_episode(reward, length, pnl, trades, fill_ratio)
        
        if self.verbose >= 1:
            n = len(self.rl_metrics.episode_rewards)
            if n % 10 == 0:
                summary = self.rl_metrics.get_summary(last_n=100)
                print(f"Episode {n}: "
                      f"mean_reward={summary['mean_reward']:.2f}, "
                      f"mean_pnl={summary['mean_pnl']:.2f}")
    
    def _print_step_metrics(self, metrics: Dict[str, float]) -> None:
        """Print step metrics to console."""
        step = metrics.get('step', 0)
        reward = metrics.get('reward', 0)
        print(f"Step {step}: reward={reward:.4f}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary of logged metrics."""
        return {
            'total_steps': self.current_step,
            'total_episodes': len(self.rl_metrics.episode_rewards),
            'episode_stats': self.rl_metrics.get_summary(),
        }
    
    def save(self) -> None:
        """Save all logged metrics."""
        if self.log_dir is None:
            return
        
        # Save RL metrics
        self.rl_metrics.save(self.log_dir / 'rl_metrics.json')
        
        # Save step metrics
        with open(self.log_dir / 'step_metrics.json', 'w') as f:
            json.dump(self.step_metrics, f, indent=2)
        
        # Save summary
        with open(self.log_dir / 'summary.json', 'w') as f:
            json.dump(self.get_summary(), f, indent=2)


def compute_all_metrics(result: SimulationResult) -> Dict[str, float]:
    """
    Compute all available metrics from a simulation result.
    
    Args:
        result: SimulationResult object
    
    Returns:
        Dict of all metrics
    """
    metrics = result.to_dict()
    
    # Add derived metrics
    if result.pnl_history:
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(result.pnl_history)
        metrics['sortino_ratio'] = calculate_sortino_ratio(result.pnl_history)
        metrics['max_drawdown'] = calculate_max_drawdown(result.pnl_history)
        metrics['calmar_ratio'] = calculate_calmar_ratio(result.pnl_history)
    
    # Client analysis
    toxicity = analyze_client_toxicity(result)
    metrics['n_toxic_clients'] = len(toxicity['toxic'])
    metrics['n_good_clients'] = len(toxicity['good'])
    
    return metrics
