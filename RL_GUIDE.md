# Reinforcement Learning Guide

This guide covers how to use the illiquid bond market simulator for RL research.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Overview](#environment-overview)
3. [Observation Space](#observation-space)
4. [Action Space](#action-space)
5. [Reward Functions](#reward-functions)
6. [Configuration](#configuration)
7. [Training with Popular Libraries](#training-with-popular-libraries)
8. [Baseline Agents](#baseline-agents)
9. [Multi-Agent Mode](#multi-agent-mode)
10. [Offline RL](#offline-rl)
11. [Evaluation and Benchmarking](#evaluation-and-benchmarking)

## Quick Start

```python
from illiquid_market_sim import TradingEnv, EnvConfig

# Create environment
env = TradingEnv(EnvConfig(max_episode_steps=100))

# Reset and get initial observation
obs, info = env.reset(seed=42)

# Run episode
total_reward = 0
done = False

while not done:
    # Your policy here (or use random actions)
    action = env.action_space_sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode reward: {total_reward:.2f}")
print(f"Final PnL: {info.get('total_pnl', 0):.2f}")
```

## Environment Overview

The `TradingEnv` simulates a dealer in an illiquid bond market. The agent:

1. **Receives RFQs** (Request for Quotes) from various client types
2. **Quotes prices** by selecting a bid-ask spread
3. **Executes trades** when clients accept quotes
4. **Manages inventory** and risk across multiple bonds

The goal is to maximize risk-adjusted PnL while managing:
- Inventory risk (holding too much of any bond)
- Adverse selection (trading with informed clients)
- Market impact (moving prices against yourself)

## Observation Space

The observation is a 40-dimensional vector containing:

### RFQ Features (8 dims)
- `side`: 1.0 for buy, -1.0 for sell
- `size`: Normalized trade size
- `client_type_*`: One-hot encoding of client type (real_money, hedge_fund, fisher, noise)
- `time_in_episode`: Normalized step number
- `rfq_count`: Total RFQs received

### Bond Features (10 dims)
- `sector_*`: One-hot encoding of sector (IG, HY, EM)
- `liquidity`: Bond liquidity score [0, 1]
- `volatility`: Normalized volatility
- `maturity`: Normalized years to maturity
- `base_spread`: Normalized base spread
- `last_price`: Last traded price
- `naive_mid`: Dealer's mid estimate
- `current_position`: Current position in this bond

### Portfolio Features (6 dims)
- `total_pnl`: Normalized total PnL
- `realized_pnl`: Normalized realized PnL
- `inventory_risk`: Total absolute inventory
- `num_positions`: Fraction of bonds with positions
- `total_trades`: Normalized trade count
- `fill_ratio`: RFQ fill ratio

### Market Features (6 dims)
- `level_IG/HY/EM`: Sector factor levels
- `time_in_episode`: Episode progress
- `recent_volatility`: Recent PnL volatility
- `regime_stress`: Market stress indicator

### History Features (10 dims)
- Recent PnL values
- Last trade information
- Episode progress and cumulative reward

```python
from illiquid_market_sim import get_observation_spec

spec = get_observation_spec()
print(spec.describe())
```

## Action Space

### Continuous Spread (default)
Single float in [0, 1] mapped to [min_spread_bps, max_spread_bps]:

```python
config = EnvConfig(
    action_type=ActionType.CONTINUOUS_SPREAD,
    min_spread_bps=10.0,
    max_spread_bps=200.0,
)
```

### Discrete Spread
Integer selecting from N spread tiers:

```python
config = EnvConfig(
    action_type=ActionType.DISCRETE_SPREAD,
    discrete_spread_levels=10,
)
```

### Accept/Reject
Two-dimensional: (accept decision, spread if accepting):

```python
config = EnvConfig(action_type=ActionType.ACCEPT_REJECT)
```

## Reward Functions

### PnL (Simple)
Raw PnL change from previous step:
```python
config = EnvConfig(reward_type=RewardType.PNL)
```

### Risk-Adjusted PnL (Recommended)
PnL minus inventory penalty:
```python
config = EnvConfig(
    reward_type=RewardType.RISK_ADJUSTED_PNL,
    inventory_penalty=0.01,
)
```

### Execution Quality
Edge captured vs fair value:
```python
config = EnvConfig(reward_type=RewardType.EXECUTION_QUALITY)
```

### Sharpe-like
PnL normalized by rolling volatility:
```python
config = EnvConfig(reward_type=RewardType.SHARPE)
```

### Composite
Weighted combination of above:
```python
config = EnvConfig(reward_type=RewardType.COMPOSITE)
```

## Configuration

### Environment Configuration

```python
from illiquid_market_sim import EnvConfig, SimulationConfig

env_config = EnvConfig(
    # Episode settings
    max_episode_steps=100,
    
    # Reward settings
    reward_type=RewardType.RISK_ADJUSTED_PNL,
    reward_scale=1.0,
    inventory_penalty=0.01,
    
    # Action settings
    action_type=ActionType.CONTINUOUS_SPREAD,
    min_spread_bps=10.0,
    max_spread_bps=200.0,
    
    # Observation settings
    normalize_obs=True,
    normalize_reward=False,
    
    # Simulation config
    sim_config=SimulationConfig(
        num_bonds=50,
        num_steps=100,
        market_volatility=0.02,
        jump_probability=0.05,
    ),
    
    # Reproducibility
    seed=42,
)
```

### Market Regime Presets

```python
from illiquid_market_sim import get_preset, list_presets

# List available presets
print(list_presets())
# ['default', 'normal', 'stressed', 'crisis', 'liquid', 'illiquid', 'rl_easy', 'rl_medium', 'rl_hard']

# Use a preset
sim_config = get_preset("rl_easy")
env = TradingEnv(EnvConfig(sim_config=sim_config))
```

## Training with Popular Libraries

### Stable-Baselines3

```python
from stable_baselines3 import PPO
from illiquid_market_sim import TradingEnv, EnvConfig
from illiquid_market_sim.rl import make_sb3_env

# Create wrapped environment
env = make_sb3_env(EnvConfig())

# Train
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save
model.save("ppo_trading")
```

### CleanRL Style

```python
from illiquid_market_sim import TradingEnv, EnvConfig
from illiquid_market_sim.rl import RolloutBuffer, collect_rollout

env = TradingEnv(EnvConfig())
buffer = RolloutBuffer(buffer_size=2048, obs_dim=40, action_dim=1)

# Collect rollout
obs, _ = env.reset()
for step in range(2048):
    action = your_policy(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    value = your_value_function(obs)
    log_prob = your_log_prob(obs, action)
    
    buffer.add(obs, action, reward, value, log_prob, terminated or truncated)
    
    obs = next_obs
    if terminated or truncated:
        obs, _ = env.reset()

# Compute advantages
buffer.compute_returns_and_advantages(last_value=0, gamma=0.99, gae_lambda=0.95)

# Train on buffer
for batch in buffer.get_batches(batch_size=64):
    # Your PPO update here
    pass
```

## Baseline Agents

```python
from illiquid_market_sim import TradingEnv, EnvConfig
from illiquid_market_sim.baselines import (
    RandomAgent,
    FixedSpreadAgent,
    InventoryAwareAgent,
    AdaptiveAgent,
    ConservativeAgent,
    AggressiveAgent,
    ClientTieringAgent,
)

env = TradingEnv(EnvConfig())

# Create agents
agents = {
    "random": RandomAgent(),
    "fixed_50": FixedSpreadAgent(0.5),
    "inventory_aware": InventoryAwareAgent(),
    "adaptive": AdaptiveAgent(),
    "conservative": ConservativeAgent(),
    "aggressive": AggressiveAgent(),
    "client_tiering": ClientTieringAgent(),
}

# Evaluate
from illiquid_market_sim.rl import evaluate_policy

for name, agent in agents.items():
    result = evaluate_policy(env, agent.get_policy(), n_episodes=10)
    print(f"{name}: mean_reward={result.mean_reward:.2f}")
```

### Running Benchmarks

```python
from illiquid_market_sim import TradingEnv, EnvConfig
from illiquid_market_sim.baselines import run_benchmark, get_all_baseline_agents

env = TradingEnv(EnvConfig())
agents = get_all_baseline_agents()

results = run_benchmark(env, agents, verbose=True)
```

## Multi-Agent Mode

For competitive market making scenarios:

```python
from illiquid_market_sim.multi_agent_env import (
    MultiAgentTradingEnv,
    MultiAgentEnvConfig,
    CompetitionMode,
)

config = MultiAgentEnvConfig(
    num_agents=3,
    competition_mode=CompetitionMode.WINNER_TAKES_ALL,
    max_episode_steps=100,
)

env = MultiAgentTradingEnv(config)

# PettingZoo-style API
observations, infos = env.reset()

while env.agents:
    actions = {
        agent: your_policy(observations[agent])
        for agent in env.agents
    }
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

## Offline RL

### Collecting Data

```python
from illiquid_market_sim import TradingEnv, EnvConfig
from illiquid_market_sim.rl import DataCollector

env = TradingEnv(EnvConfig())
collector = DataCollector(env)

# Collect with a policy
def expert_policy(obs):
    # Your expert policy
    return np.array([0.5])

collector.collect_episodes(
    n_episodes=1000,
    policy=expert_policy,
    verbose=True,
)

# Save dataset
collector.save("data/expert_data.npz", metadata={
    "collection_policy": "expert",
    "description": "Expert demonstrations",
})
```

### Loading and Using Data

```python
from illiquid_market_sim.rl import OfflineDataset

# Load dataset
dataset = OfflineDataset.load("data/expert_data.npz")

print(f"Dataset has {len(dataset)} transitions")
print(f"Statistics: {dataset.get_statistics()}")

# Train offline RL
for batch in dataset.iterate_batches(batch_size=256):
    obs, actions, rewards, next_obs, dones = batch
    # Your offline RL update here
```

## Evaluation and Benchmarking

### Single Policy Evaluation

```python
from illiquid_market_sim.rl import evaluate_policy

result = evaluate_policy(
    env=env,
    policy=your_policy,
    n_episodes=100,
    deterministic=True,
    seed=42,
)

print(result.summary())
```

### Comparing Multiple Policies

```python
from illiquid_market_sim.rl import benchmark_agent

agents = {
    "your_policy": your_policy,
    "baseline": baseline_policy,
}

result = benchmark_agent(env, agents, n_episodes=50)
print(result.summary())
```

### Metrics

```python
from illiquid_market_sim.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)

# From simulation result
sharpe = calculate_sharpe_ratio(result.pnl_history)
sortino = calculate_sortino_ratio(result.pnl_history)
max_dd = calculate_max_drawdown(result.pnl_history)
calmar = calculate_calmar_ratio(result.pnl_history)
```

## Tips for Training

1. **Start with RL-easy preset**: Use `get_preset("rl_easy")` for initial experiments

2. **Normalize observations**: Keep `normalize_obs=True` for stable training

3. **Use risk-adjusted rewards**: `RISK_ADJUSTED_PNL` encourages inventory management

4. **Monitor fill ratio**: Low fill ratio may indicate spreads are too wide

5. **Check for adverse selection**: Track PnL by client type

6. **Use curriculum learning**: Start with liquid markets, progress to illiquid

7. **Seed everything**: Set seeds for reproducibility

8. **Log extensively**: Use `MetricsLogger` to track training progress

## Common Issues

**Q: Agent always quotes wide spreads**
A: Try reducing `inventory_penalty` or using `EXECUTION_QUALITY` reward

**Q: Agent loses money consistently**
A: Check if agent is trading with informed clients; consider `ClientTieringAgent` as baseline

**Q: Training is unstable**
A: Enable observation/reward normalization, reduce learning rate

**Q: Episode ends too quickly**
A: Increase `max_episode_steps` or reduce `rfq_prob_per_client`

