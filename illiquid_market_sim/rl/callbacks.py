"""
Training callbacks for RL experiments.

This module provides callback classes that can be used during training
to log metrics, save checkpoints, and run evaluations.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from illiquid_market_sim.rl.evaluation import evaluate_policy, EvaluationResult


class BaseCallback:
    """Base class for training callbacks."""
    
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.n_calls = 0
        self.model = None
        self.env = None
    
    def init_callback(self, model: Any, env: Any) -> None:
        """Initialize callback with model and env references."""
        self.model = model
        self.env = env
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """Called at the start of training."""
        pass
    
    def on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            True to continue training, False to stop
        """
        self.n_calls += 1
        return True
    
    def on_rollout_start(self) -> None:
        """Called at the start of a rollout."""
        pass
    
    def on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        pass
    
    def on_training_end(self) -> None:
        """Called at the end of training."""
        pass


class EvalCallback(BaseCallback):
    """
    Callback for evaluating the policy during training.
    
    Runs evaluation episodes at regular intervals and logs results.
    Optionally saves the best model.
    
    Example:
        >>> eval_callback = EvalCallback(
        ...     eval_env=eval_env,
        ...     eval_freq=1000,
        ...     n_eval_episodes=5,
        ...     best_model_save_path="./best_model",
        ... )
    """
    
    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        callback_on_new_best: Optional[Callable] = None,
    ):
        """
        Initialize evaluation callback.
        
        Args:
            eval_env: Environment for evaluation
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of episodes per evaluation
            deterministic: Use deterministic actions
            render: Render during evaluation
            verbose: Verbosity level
            best_model_save_path: Path to save best model
            log_path: Path to save evaluation logs
            callback_on_new_best: Callback when new best is found
        """
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.callback_on_new_best = callback_on_new_best
        
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.evaluations: List[EvaluationResult] = []
        self.evaluation_timesteps: List[int] = []
    
    def on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True
    
    def _run_evaluation(self) -> None:
        """Run evaluation and log results."""
        # Get policy from model
        if hasattr(self.model, "predict"):
            policy = lambda obs: self.model.predict(obs, deterministic=self.deterministic)[0]
        elif hasattr(self.model, "get_action"):
            policy = lambda obs: self.model.get_action(obs)
        else:
            # Assume model is callable
            policy = self.model
        
        result = evaluate_policy(
            env=self.eval_env,
            policy=policy,
            n_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            render=self.render,
        )
        
        self.evaluations.append(result)
        self.evaluation_timesteps.append(self.n_calls)
        self.last_mean_reward = result.mean_reward
        
        if self.verbose > 0:
            print(f"Eval @ step {self.n_calls}: "
                  f"mean_reward={result.mean_reward:.2f} (+/- {result.std_reward:.2f})")
        
        # Check for new best
        if result.mean_reward > self.best_mean_reward:
            self.best_mean_reward = result.mean_reward
            
            if self.verbose > 0:
                print(f"New best mean reward: {self.best_mean_reward:.2f}")
            
            if self.best_model_save_path is not None:
                self._save_best_model()
            
            if self.callback_on_new_best is not None:
                self.callback_on_new_best(result)
        
        # Save logs
        if self.log_path is not None:
            self._save_logs()
    
    def _save_best_model(self) -> None:
        """Save the current model as best."""
        path = Path(self.best_model_save_path)
        path.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, "save"):
            self.model.save(path / "best_model")
        
        # Save evaluation info
        info = {
            "timestep": self.n_calls,
            "mean_reward": float(self.best_mean_reward),
        }
        with open(path / "best_model_info.json", "w") as f:
            json.dump(info, f, indent=2)
    
    def _save_logs(self) -> None:
        """Save evaluation logs."""
        path = Path(self.log_path)
        path.mkdir(parents=True, exist_ok=True)
        
        logs = {
            "timesteps": self.evaluation_timesteps,
            "mean_rewards": [r.mean_reward for r in self.evaluations],
            "std_rewards": [r.std_reward for r in self.evaluations],
        }
        
        with open(path / "evaluations.json", "w") as f:
            json.dump(logs, f, indent=2)


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints during training.
    
    Saves the model at regular intervals.
    """
    
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./checkpoints",
        name_prefix: str = "model",
        verbose: int = 1,
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_freq: Save every N steps
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint files
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()
        return True
    
    def _save_checkpoint(self) -> None:
        """Save current model checkpoint."""
        checkpoint_path = self.save_path / f"{self.name_prefix}_{self.n_calls}"
        
        if hasattr(self.model, "save"):
            self.model.save(checkpoint_path)
        
        if self.verbose > 0:
            print(f"Saved checkpoint: {checkpoint_path}")


class LoggingCallback(BaseCallback):
    """
    Callback for logging training metrics.
    
    Logs metrics to console and optionally to file.
    """
    
    def __init__(
        self,
        log_freq: int = 100,
        log_path: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Initialize logging callback.
        
        Args:
            log_freq: Log every N steps
            log_path: Path to save logs (optional)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.log_freq = log_freq
        self.log_path = Path(log_path) if log_path else None
        
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        self.start_time = None
        self.logs: List[Dict[str, Any]] = []
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        self.start_time = time.time()
        
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)
    
    def on_step(self) -> bool:
        self.current_episode_length += 1
        
        # Note: In practice, you'd extract reward from the step
        # This is a simplified version
        
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()
        
        return True
    
    def on_rollout_end(self) -> None:
        """Record episode statistics."""
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
    
    def _log_metrics(self) -> None:
        """Log current metrics."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        fps = self.n_calls / elapsed_time if elapsed_time > 0 else 0
        
        metrics = {
            "timestep": self.n_calls,
            "elapsed_time": elapsed_time,
            "fps": fps,
        }
        
        if self.episode_rewards:
            metrics["mean_episode_reward"] = np.mean(self.episode_rewards[-100:])
            metrics["mean_episode_length"] = np.mean(self.episode_lengths[-100:])
        
        self.logs.append(metrics)
        
        if self.verbose > 0:
            print(f"Step {self.n_calls}: "
                  f"fps={fps:.1f}, "
                  f"mean_reward={metrics.get('mean_episode_reward', 0):.2f}")
        
        if self.log_path:
            self._save_logs()
    
    def _save_logs(self) -> None:
        """Save logs to file."""
        with open(self.log_path / "training_log.json", "w") as f:
            json.dump(self.logs, f, indent=2)
    
    def on_training_end(self) -> None:
        """Final logging."""
        if self.log_path:
            self._save_logs()
        
        if self.verbose > 0:
            total_time = time.time() - self.start_time if self.start_time else 0
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Total steps: {self.n_calls}")
            if self.episode_rewards:
                print(f"Final mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")


class CallbackList(BaseCallback):
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: List[BaseCallback]):
        super().__init__()
        self.callbacks = callbacks
    
    def init_callback(self, model: Any, env: Any) -> None:
        for callback in self.callbacks:
            callback.init_callback(model, env)
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_training_start(locals_, globals_)
    
    def on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            if not callback.on_step():
                continue_training = False
        return continue_training
    
    def on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()
    
    def on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()
    
    def on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

