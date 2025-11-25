"""
Illiquid Bond Market Simulator

A self-contained simulation of an illiquid bond market with:
- Synthetic bonds with hidden fair values
- Multiple client types with different behaviors
- Dealer quoting logic
- Market impact modeling
- P&L tracking
- Gymnasium-compatible RL environment

Quick Start (RL):
    >>> from illiquid_market_sim import TradingEnv, EnvConfig
    >>> env = TradingEnv(EnvConfig(max_episode_steps=100))
    >>> obs, info = env.reset(seed=42)
    >>> action = env.action_space_sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)

Quick Start (Simulation):
    >>> from illiquid_market_sim import Simulator, SimulationConfig
    >>> sim = Simulator(SimulationConfig(num_steps=100))
    >>> result = sim.run(verbose=True)
"""

__version__ = "0.2.0"

# Core simulation components
from illiquid_market_sim.bonds import Bond, generate_bond_universe
from illiquid_market_sim.clients import (
    Client,
    RealMoneyClient,
    HedgeFundClient,
    FisherClient,
    NoiseClient,
    create_client_universe,
)
from illiquid_market_sim.config import (
    SimulationConfig,
    MarketRegime,
    BondUniverseConfig,
    ClientConfig,
    MarketDynamicsConfig,
    DealerConfig,
    get_regime_config,
    get_preset,
    list_presets,
    PRESET_CONFIGS,
    DEFAULT_CONFIG,
)
from illiquid_market_sim.market import MarketState, MarketImpactModel
from illiquid_market_sim.portfolio import Portfolio, Position
from illiquid_market_sim.rfq import RFQ, Quote, Trade
from illiquid_market_sim.agent import (
    DealerAgent,
    QuotingStrategy,
    RuleBasedQuotingStrategy,
    ManualQuotingStrategy,
)
from illiquid_market_sim.simulation import Simulator
from illiquid_market_sim.metrics import (
    SimulationResult,
    calculate_impact_cost,
    summarize_simulation,
    analyze_client_toxicity,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

# RL environment
from illiquid_market_sim.env import (
    TradingEnv,
    EnvConfig,
    RewardType,
    ActionType,
    ObservationSpace,
    ActionSpace,
    RLQuotingStrategy,
    make_gymnasium_env,
)

# Multi-agent environment
from illiquid_market_sim.multi_agent_env import (
    MultiAgentTradingEnv,
    MultiAgentEnvConfig,
    CompetitionMode,
    DealerState,
    make_multi_agent_env,
)

# Spaces and normalization
from illiquid_market_sim.spaces import (
    ObservationSpec,
    ActionSpec,
    RewardSpec,
    FeatureGroup,
    get_observation_spec,
    get_action_spec,
    get_reward_spec,
    decode_action,
    encode_action,
    RunningNormalizer,
    ObservationNormalizer,
    RewardNormalizer,
    validate_observation,
    validate_action,
)

# Baselines
from illiquid_market_sim.baselines import (
    BaselineAgent,
    RandomAgent,
    FixedSpreadAgent,
    InventoryAwareAgent,
    AdaptiveAgent,
    ConservativeAgent,
    AggressiveAgent,
    TrendFollowingAgent,
    ClientTieringAgent,
    BenchmarkTask,
    BENCHMARK_TASKS,
    get_all_baseline_agents,
    run_benchmark,
)

# Advanced features
from illiquid_market_sim.advanced import (
    CurriculumStage,
    CurriculumConfig,
    CurriculumScheduler,
    StressScenario,
    StressEvent,
    StressGenerator,
    ObjectiveType,
    MultiObjectiveConfig,
    MultiObjectiveReward,
    DomainRandomizationConfig,
    DomainRandomizer,
)

__all__ = [
    # Version
    "__version__",
    
    # Bonds
    "Bond",
    "generate_bond_universe",
    
    # Clients
    "Client",
    "RealMoneyClient",
    "HedgeFundClient",
    "FisherClient",
    "NoiseClient",
    "create_client_universe",
    
    # Configuration
    "SimulationConfig",
    "MarketRegime",
    "BondUniverseConfig",
    "ClientConfig",
    "MarketDynamicsConfig",
    "DealerConfig",
    "get_regime_config",
    "get_preset",
    "list_presets",
    "PRESET_CONFIGS",
    "DEFAULT_CONFIG",
    
    # Market
    "MarketState",
    "MarketImpactModel",
    
    # Portfolio
    "Portfolio",
    "Position",
    
    # RFQ
    "RFQ",
    "Quote",
    "Trade",
    
    # Agent
    "DealerAgent",
    "QuotingStrategy",
    "RuleBasedQuotingStrategy",
    "ManualQuotingStrategy",
    
    # Simulation
    "Simulator",
    
    # Metrics
    "SimulationResult",
    "calculate_impact_cost",
    "summarize_simulation",
    "analyze_client_toxicity",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    
    # RL Environment
    "TradingEnv",
    "EnvConfig",
    "RewardType",
    "ActionType",
    "ObservationSpace",
    "ActionSpace",
    "RLQuotingStrategy",
    "make_gymnasium_env",
    
    # Multi-Agent Environment
    "MultiAgentTradingEnv",
    "MultiAgentEnvConfig",
    "CompetitionMode",
    "DealerState",
    "make_multi_agent_env",
    
    # Spaces and Normalization
    "ObservationSpec",
    "ActionSpec",
    "RewardSpec",
    "FeatureGroup",
    "get_observation_spec",
    "get_action_spec",
    "get_reward_spec",
    "decode_action",
    "encode_action",
    "RunningNormalizer",
    "ObservationNormalizer",
    "RewardNormalizer",
    "validate_observation",
    "validate_action",
    
    # Baselines
    "BaselineAgent",
    "RandomAgent",
    "FixedSpreadAgent",
    "InventoryAwareAgent",
    "AdaptiveAgent",
    "ConservativeAgent",
    "AggressiveAgent",
    "TrendFollowingAgent",
    "ClientTieringAgent",
    "BenchmarkTask",
    "BENCHMARK_TASKS",
    "get_all_baseline_agents",
    "run_benchmark",
    
    # Advanced Features
    "CurriculumStage",
    "CurriculumConfig",
    "CurriculumScheduler",
    "StressScenario",
    "StressEvent",
    "StressGenerator",
    "ObjectiveType",
    "MultiObjectiveConfig",
    "MultiObjectiveReward",
    "DomainRandomizationConfig",
    "DomainRandomizer",
]
