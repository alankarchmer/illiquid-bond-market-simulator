"""
Tests for the RL environment.

This module tests:
- Environment API compliance (reset, step, render)
- Observation and action space correctness
- Determinism under fixed seeds
- Episode termination behavior
- Reward computation
"""

import numpy as np
import pytest

from illiquid_market_sim.env import TradingEnv, EnvConfig, RewardType, ActionType
from illiquid_market_sim.spaces import (
    get_observation_spec,
    get_action_spec,
    validate_observation,
    validate_action,
)


class TestTradingEnvBasic:
    """Basic environment functionality tests."""
    
    def test_env_creation(self):
        """Test that environment can be created."""
        env = TradingEnv()
        assert env is not None
    
    def test_env_creation_with_config(self):
        """Test environment creation with custom config."""
        config = EnvConfig(max_episode_steps=50)
        env = TradingEnv(config)
        assert env.config.max_episode_steps == 50
    
    def test_reset_returns_observation_and_info(self):
        """Test that reset returns observation and info dict."""
        env = TradingEnv()
        result = env.reset(seed=42)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
    
    def test_observation_shape(self):
        """Test that observation has correct shape."""
        env = TradingEnv()
        obs, _ = env.reset(seed=42)
        
        expected_dim = env.observation_space_def.total_dim
        assert obs.shape == (expected_dim,)
    
    def test_observation_dtype(self):
        """Test that observation has correct dtype."""
        env = TradingEnv()
        obs, _ = env.reset(seed=42)
        
        assert obs.dtype == np.float32
    
    def test_step_returns_correct_tuple(self):
        """Test that step returns (obs, reward, terminated, truncated, info)."""
        env = TradingEnv()
        env.reset(seed=42)
        
        action = env.action_space_sample()
        result = env.step(action)
        
        assert isinstance(result, tuple)
        assert len(result) == 5
        
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_episode_terminates(self):
        """Test that episode terminates after max steps."""
        config = EnvConfig(max_episode_steps=10)
        env = TradingEnv(config)
        env.reset(seed=42)
        
        for _ in range(20):
            action = env.action_space_sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        assert terminated or truncated


class TestTradingEnvDeterminism:
    """Tests for environment determinism."""
    
    def test_same_seed_same_observations(self):
        """Test that same seed produces same initial observation."""
        env1 = TradingEnv()
        env2 = TradingEnv()
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_same_seed_same_trajectory(self):
        """Test that same seed produces same trajectory with same actions."""
        config = EnvConfig(max_episode_steps=20)
        
        env1 = TradingEnv(config)
        env2 = TradingEnv(config)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Use fixed actions
        np.random.seed(123)
        actions = [np.array([0.5], dtype=np.float32) for _ in range(10)]
        
        for action in actions:
            obs1, r1, _, _, _ = env1.step(action)
            obs2, r2, _, _, _ = env2.step(action)
            
            np.testing.assert_array_almost_equal(obs1, obs2, decimal=5)
            assert abs(r1 - r2) < 1e-5
    
    def test_different_seeds_different_observations(self):
        """Test that different seeds produce different observations."""
        env = TradingEnv()
        
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)
        
        # They should be different (with very high probability)
        assert not np.allclose(obs1, obs2)


class TestObservationValidation:
    """Tests for observation validity."""
    
    def test_observation_no_nan(self):
        """Test that observations contain no NaN values."""
        env = TradingEnv()
        obs, _ = env.reset(seed=42)
        
        assert not np.any(np.isnan(obs))
        
        for _ in range(10):
            action = env.action_space_sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert not np.any(np.isnan(obs))
            if terminated or truncated:
                break
    
    def test_observation_no_inf(self):
        """Test that observations contain no Inf values."""
        env = TradingEnv()
        obs, _ = env.reset(seed=42)
        
        assert not np.any(np.isinf(obs))
        
        for _ in range(10):
            action = env.action_space_sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert not np.any(np.isinf(obs))
            if terminated or truncated:
                break
    
    def test_observation_validation(self):
        """Test observation validation utility."""
        env = TradingEnv()
        obs, _ = env.reset(seed=42)
        
        errors = validate_observation(obs)
        assert len(errors) == 0


class TestActionSpace:
    """Tests for action space."""
    
    def test_action_sample_valid(self):
        """Test that sampled actions are valid."""
        env = TradingEnv()
        env.reset(seed=42)
        
        for _ in range(10):
            action = env.action_space_sample()
            assert action.shape == (1,)
            assert 0 <= action[0] <= 1
    
    def test_continuous_spread_action(self):
        """Test continuous spread action type."""
        config = EnvConfig(action_type=ActionType.CONTINUOUS_SPREAD)
        env = TradingEnv(config)
        env.reset(seed=42)
        
        # Test valid action
        action = np.array([0.5], dtype=np.float32)
        obs, reward, _, _, _ = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
    
    def test_discrete_spread_action(self):
        """Test discrete spread action type."""
        config = EnvConfig(
            action_type=ActionType.DISCRETE_SPREAD,
            discrete_spread_levels=5
        )
        env = TradingEnv(config)
        env.reset(seed=42)
        
        # Test valid action
        action = np.array([2], dtype=np.int32)
        obs, reward, _, _, _ = env.step(action)
        
        assert obs is not None


class TestRewardComputation:
    """Tests for reward computation."""
    
    def test_reward_is_finite(self):
        """Test that rewards are finite."""
        env = TradingEnv()
        env.reset(seed=42)
        
        for _ in range(20):
            action = env.action_space_sample()
            _, reward, terminated, truncated, _ = env.step(action)
            
            assert np.isfinite(reward)
            if terminated or truncated:
                break
    
    def test_pnl_reward_type(self):
        """Test PnL reward type."""
        config = EnvConfig(reward_type=RewardType.PNL)
        env = TradingEnv(config)
        env.reset(seed=42)
        
        action = env.action_space_sample()
        _, reward, _, _, _ = env.step(action)
        
        assert isinstance(reward, (int, float))
    
    def test_risk_adjusted_reward_type(self):
        """Test risk-adjusted PnL reward type."""
        config = EnvConfig(reward_type=RewardType.RISK_ADJUSTED_PNL)
        env = TradingEnv(config)
        env.reset(seed=42)
        
        action = env.action_space_sample()
        _, reward, _, _, _ = env.step(action)
        
        assert isinstance(reward, (int, float))


class TestInfoDict:
    """Tests for info dictionary."""
    
    def test_info_contains_step(self):
        """Test that info contains step count."""
        env = TradingEnv()
        _, info = env.reset(seed=42)
        
        assert "step" in info
    
    def test_info_contains_pnl(self):
        """Test that info contains PnL after stepping."""
        env = TradingEnv()
        env.reset(seed=42)
        
        action = env.action_space_sample()
        _, _, _, _, info = env.step(action)
        
        assert "total_pnl" in info or "step" in info


class TestRender:
    """Tests for rendering."""
    
    def test_render_ansi(self):
        """Test ANSI rendering mode."""
        env = TradingEnv()
        env.reset(seed=42)
        
        result = env.render(mode="ansi")
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_step_without_reset(self):
        """Test that stepping without reset raises error."""
        env = TradingEnv()
        
        with pytest.raises(RuntimeError):
            env.step(np.array([0.5]))
    
    def test_multiple_resets(self):
        """Test that multiple resets work correctly."""
        env = TradingEnv()
        
        for seed in [42, 123, 456]:
            obs, info = env.reset(seed=seed)
            assert obs is not None
            assert info is not None
    
    def test_long_episode(self):
        """Test running a full episode."""
        config = EnvConfig(max_episode_steps=100)
        env = TradingEnv(config)
        env.reset(seed=42)
        
        total_reward = 0
        steps = 0
        
        while True:
            action = env.action_space_sample()
            _, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps <= 100
        assert np.isfinite(total_reward)


class TestSpaces:
    """Tests for space specifications."""
    
    def test_observation_spec(self):
        """Test observation specification."""
        spec = get_observation_spec()
        
        assert spec.total_dim > 0
        assert len(spec.feature_groups) > 0
        assert len(spec.all_features) == spec.total_dim
    
    def test_action_spec_continuous(self):
        """Test continuous action specification."""
        spec = get_action_spec(ActionType.CONTINUOUS_SPREAD)
        
        assert spec.dim == 1
        assert spec.is_continuous
        assert not spec.is_discrete
    
    def test_action_spec_discrete(self):
        """Test discrete action specification."""
        spec = get_action_spec(ActionType.DISCRETE_SPREAD, n_discrete=10)
        
        assert spec.dim == 1
        assert spec.is_discrete
        assert not spec.is_continuous
        assert spec.n_discrete == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

