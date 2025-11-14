"""
Simulation orchestration - ties together all components.
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass

from illiquid_market_sim.config import SimulationConfig
from illiquid_market_sim.bonds import Bond, generate_bond_universe
from illiquid_market_sim.clients import Client, create_client_universe, create_clients_from_scenario
from illiquid_market_sim.market import MarketState, MarketImpactModel
from illiquid_market_sim.portfolio import Portfolio
from illiquid_market_sim.agent import DealerAgent, QuotingStrategy
from illiquid_market_sim.rfq import RFQ, Quote, Trade
from illiquid_market_sim.metrics import (
    SimulationResult, 
    calculate_impact_cost,
    summarize_simulation
)
from illiquid_market_sim.scenarios import ScenarioConfig


class Simulator:
    """
    Orchestrates the full simulation of the illiquid bond market.
    
    The simulation loop:
    1. Advance market state (factor evolution, potential jumps)
    2. Clients generate RFQs
    3. Dealer quotes each RFQ
    4. Clients decide whether to trade
    5. Execute trades and apply market impact
    6. Mark portfolio to market
    7. Record metrics
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        custom_quoting_strategy: Optional[QuotingStrategy] = None,
        scenario: Optional[ScenarioConfig] = None
    ):
        """
        Initialize simulator.
        
        Args:
            config: Simulation configuration (legacy support)
            custom_quoting_strategy: Optional custom quoting strategy
            scenario: Scenario configuration (overrides config if provided)
        """
        self.config = config or SimulationConfig()
        self.scenario = scenario
        
        # Set random seed
        random.seed(self.config.random_seed)
        
        # Initialize components
        self.bonds = generate_bond_universe(
            self.config.num_bonds,
            seed=self.config.random_seed
        )
        
        # Create clients based on scenario or config
        if scenario:
            client_counts = scenario.get_total_client_count(base_count=10)
            # Calculate RFQ probability multiplier from scenario's avg_rfq_per_step
            # Baseline: avg 1 RFQ per step with 10 clients at 0.1 prob each = 1 RFQ
            # So multiplier = avg_rfq_per_step / 1.0
            rfq_prob_mult = max(0.01, scenario.avg_rfq_per_step / 1.0)
            self.clients = create_clients_from_scenario(
                client_counts=client_counts,
                rfq_probability_multiplier=rfq_prob_mult,
                size_multiplier=scenario.rfq_size_multiplier
            )
            # Apply buy_sell_skew to all clients
            for client in self.clients:
                client.buy_sell_skew = scenario.buy_sell_skew
        else:
            self.clients = create_client_universe(
                num_real_money=self.config.num_real_money_clients,
                num_hedge_fund=self.config.num_hedge_fund_clients,
                num_fisher=self.config.num_fisher_clients,
                num_noise=self.config.num_noise_clients
            )
        
        # Initialize market state with scenario parameters
        if scenario:
            self.market_state = MarketState(
                volatility=scenario.base_volatility,
                jump_probability=scenario.market_shock_prob,
                spread_drift_bps=scenario.spread_drift_bps_per_step,
                sector_shock_prob=scenario.sector_shock_prob,
                idiosyncratic_event_prob=scenario.idiosyncratic_event_prob
            )
        else:
            self.market_state = MarketState(
                volatility=self.config.market_volatility,
                jump_probability=self.config.jump_probability
            )
        
        # Initialize impact model with scenario parameters
        if scenario:
            self.impact_model = MarketImpactModel(
                base_impact_coeff=scenario.impact_coefficient,
                cross_impact_factor=self.config.cross_impact_factor,  # Legacy
                impact_decay=self.config.impact_decay,
                liquidity_multiplier=scenario.liquidity_multiplier,
                impact_cross_issuer=scenario.impact_cross_issuer,
                impact_cross_sector=scenario.impact_cross_sector
            )
        else:
            self.impact_model = MarketImpactModel(
                base_impact_coeff=self.config.base_impact_coeff,
                cross_impact_factor=self.config.cross_impact_factor,
                impact_decay=self.config.impact_decay
            )
        
        self.portfolio = Portfolio()
        
        # Apply initial positions if specified in scenario
        if scenario and scenario.initial_positions:
            self._apply_initial_positions(scenario.initial_positions)
        
        self.dealer = DealerAgent(
            portfolio=self.portfolio,
            quoting_strategy=custom_quoting_strategy
        )
        
        # Tracking
        self.all_rfqs: List[RFQ] = []
        self.all_quotes: List[Quote] = []
        self.all_trades: List[Trade] = []
        self.events: List[Dict] = []
        
        self._trade_counter = 0
        self._bonds_dict = {b.id: b for b in self.bonds}
        
        # Store base client parameters for regime shifts
        # (to avoid exponential growth when parameters are updated)
        self._client_base_params = {
            client.client_id: {
                'mean_size': client.mean_size,
                'rfq_probability': client.rfq_probability
            }
            for client in self.clients
        }
    
    def run(
        self,
        num_steps: Optional[int] = None,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            num_steps: Number of timesteps (uses config default if None)
            verbose: Whether to print progress
            
        Returns:
            SimulationResult with all metrics
        """
        steps = num_steps or self.config.num_steps
        
        if verbose:
            print(f"Starting simulation: {steps} steps, {len(self.bonds)} bonds, "
                  f"{len(self.clients)} clients")
            print("=" * 70)
        
        for step in range(steps):
            if verbose and step % 10 == 0:
                print(f"Step {step}/{steps}...")
            
            self.run_step(step, verbose=verbose)
        
        # Final mark to market
        final_mtm = self.portfolio.mark_to_market(self.bonds)
        
        # Compile results
        result = self._compile_results(steps, final_mtm)
        
        if verbose:
            print("\n" + summarize_simulation(result))
        
        return result
    
    def run_step(self, step_idx: int, verbose: bool = False) -> None:
        """
        Execute one timestep of the simulation.
        
        Args:
            step_idx: Current step number
            verbose: Whether to print details
        """
        # Check if we need to update scenario parameters (regime shift)
        if self.scenario and self.scenario.regime_shift_callback:
            total_steps = self.config.num_steps
            new_scenario = self.scenario.regime_shift_callback(step_idx, total_steps)
            self._update_scenario_parameters(new_scenario)
        
        # 1. Advance market state
        event = self.market_state.step(self.bonds)
        if event:
            self.events.append({'step': step_idx, **event})
            if verbose:
                print(f"  [Event] {event['type']}: {event}")
        
        # 2. Generate RFQs from clients
        rfqs_this_step = []
        for client in self.clients:
            rfq = client.maybe_generate_rfq(
                timestep=step_idx,
                market_state=self.market_state,
                bonds=self.bonds
            )
            if rfq:
                rfqs_this_step.append((client, rfq))
                self.all_rfqs.append(rfq)
        
        if verbose and rfqs_this_step:
            print(f"  Received {len(rfqs_this_step)} RFQs")
        
        # 3. Process each RFQ
        for client, rfq in rfqs_this_step:
            self._process_rfq(client, rfq, step_idx, verbose)
        
        # 4. Mark to market
        mtm = self.portfolio.mark_to_market(self.bonds)
        
        if verbose and step_idx % 10 == 0:
            print(f"  MTM: Total P&L = {mtm['total_pnl']:+.2f}, "
                  f"Positions = {len([p for p in self.portfolio.positions.values() if abs(p.quantity) > 0.01])}")
    
    def _process_rfq(
        self,
        client: Client,
        rfq: RFQ,
        step_idx: int,
        verbose: bool
    ) -> None:
        """Process a single RFQ."""
        bond = self._bonds_dict.get(rfq.bond_id)
        if not bond:
            return
        
        # Generate quote
        quote = self.dealer.quote_for_rfq(rfq, bond, self.market_state)
        self.all_quotes.append(quote)
        
        # Client decides whether to trade
        fair_value_estimate = client.get_fair_value_estimate(bond, self.market_state)
        will_trade = client.decide_trade(rfq, quote, fair_value_estimate)
        
        if will_trade:
            # Execute trade
            self._execute_trade(client, rfq, quote, bond, verbose)
    
    def _execute_trade(
        self,
        client: Client,
        rfq: RFQ,
        quote: Quote,
        bond: Bond,
        verbose: bool
    ) -> None:
        """Execute a trade."""
        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:05d}"
        
        # Create trade (dealer's side is opposite of client's)
        trade = Trade.from_rfq_and_quote(rfq, quote, trade_id)
        self.all_trades.append(trade)
        
        if verbose:
            print(f"    [TRADE] {trade}")
        
        # Update portfolio
        self.portfolio.update_on_trade(
            bond_id=trade.bond_id,
            side=trade.side,
            size=trade.size,
            price=trade.price
        )
        
        # Record the trade in the bond
        bond.record_trade(quote.price)
        
        # Apply market impact
        impacts = self.impact_model.apply_trade_impact(
            traded_bond=bond,
            side=trade.side,
            size=trade.size,
            all_bonds=self.bonds
        )
        
        if verbose and abs(impacts.get(bond.id, 0)) > 0.1:
            print(f"      Impact: {impacts.get(bond.id, 0):+.2f} on {bond.id}")
        
        # Record trade for client stats
        true_fair = bond.get_true_fair_price()
        self.dealer.record_trade(rfq, quote, true_fair)
    
    def _compile_results(
        self,
        steps: int,
        final_mtm: Dict
    ) -> SimulationResult:
        """Compile all results into a SimulationResult object."""
        
        # Calculate metrics
        total_rfqs = len(self.all_rfqs)
        total_trades = len(self.all_trades)
        fill_ratio = total_trades / total_rfqs if total_rfqs > 0 else 0.0
        
        final_pnl = final_mtm['total_pnl']
        realized_pnl = final_mtm['realized_pnl']
        unrealized_pnl = final_mtm['unrealized_pnl']
        
        impact_cost = calculate_impact_cost(self.all_trades, self._bonds_dict)
        
        portfolio_stats = self.portfolio.get_summary_stats()
        inventory_risk = portfolio_stats['inventory_risk']
        
        # Client breakdown
        client_breakdown = {}
        for client in self.clients:
            stats = self.dealer.get_client_stats(client.client_id)
            if stats:
                client_breakdown[client.client_id] = {
                    'type': client.client_type,
                    'rfq_count': stats.rfq_count,
                    'trade_count': stats.trade_count,
                    'fill_ratio': stats.get_fill_ratio(),
                    'avg_edge': stats.get_avg_edge()
                }
        
        # P&L history
        pnl_history = self.portfolio.get_pnl_history()
        
        # RFQ history (simplified)
        rfq_history = [
            {
                'rfq_id': rfq.rfq_id,
                'timestamp': rfq.timestamp,
                'client_id': rfq.client_id,
                'bond_id': rfq.bond_id,
                'side': rfq.side,
                'size': rfq.size
            }
            for rfq in self.all_rfqs
        ]
        
        result = SimulationResult(
            total_steps=steps,
            total_rfqs=total_rfqs,
            total_trades=total_trades,
            fill_ratio=fill_ratio,
            final_pnl=final_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            impact_cost=impact_cost,
            inventory_risk=inventory_risk,
            trades=self.all_trades,
            rfq_history=rfq_history,
            pnl_history=pnl_history,
            client_breakdown=client_breakdown,
            position_summary=portfolio_stats
        )
        
        return result
    
    def get_current_state_summary(self) -> Dict:
        """Get a summary of current state (useful for debugging/monitoring)."""
        mtm = self.portfolio.mark_to_market(self.bonds)
        
        return {
            'current_step': self.market_state.current_step,
            'total_rfqs': len(self.all_rfqs),
            'total_trades': len(self.all_trades),
            'current_pnl': mtm['total_pnl'],
            'num_positions': len([p for p in self.portfolio.positions.values() 
                                  if abs(p.quantity) > 0.01]),
            'inventory_risk': self.portfolio.get_inventory_risk()
        }
    
    def _apply_initial_positions(self, positions: Dict[str, float]) -> None:
        """
        Apply initial positions to the portfolio.
        
        For scenarios like short_squeeze or inventory_overhang that start
        with non-zero positions.
        
        Args:
            positions: Dictionary mapping bond_id to quantity
        """
        # For now, we'll just set initial positions directly
        # In a more sophisticated implementation, we might select specific
        # bonds based on sector/issuer criteria
        pass  # Implementation would depend on how initial_positions is structured
    
    def _update_scenario_parameters(self, new_scenario: ScenarioConfig) -> None:
        """
        Update market and client parameters based on a new scenario config.
        
        Used for regime shift scenarios.
        
        Args:
            new_scenario: New scenario configuration to apply
        """
        # Update market state parameters
        self.market_state.volatility = new_scenario.base_volatility
        self.market_state.spread_drift_bps = new_scenario.spread_drift_bps_per_step
        self.market_state.jump_probability = new_scenario.market_shock_prob
        self.market_state.sector_shock_prob = new_scenario.sector_shock_prob
        self.market_state.idiosyncratic_event_prob = new_scenario.idiosyncratic_event_prob
        
        # Update impact model parameters
        self.impact_model.base_impact_coeff = new_scenario.impact_coefficient
        self.impact_model.liquidity_multiplier = new_scenario.liquidity_multiplier
        self.impact_model.impact_cross_issuer = new_scenario.impact_cross_issuer
        self.impact_model.impact_cross_sector = new_scenario.impact_cross_sector
        
        # Update client parameters
        rfq_prob_mult = max(0.01, new_scenario.avg_rfq_per_step / 1.0)
        for client in self.clients:
            # Get base values to avoid exponential growth
            base_params = self._client_base_params.get(client.client_id, {})
            base_mean_size = base_params.get('mean_size', client.mean_size)
            base_rfq_prob = base_params.get('rfq_probability', client.rfq_probability)
            
            # Set values from base * multiplier (not *= which compounds)
            client.buy_sell_skew = new_scenario.buy_sell_skew
            client.mean_size = base_mean_size * new_scenario.rfq_size_multiplier
            client.rfq_probability = base_rfq_prob * rfq_prob_mult
