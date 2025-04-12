import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple, Any, Optional

# Assuming PongActorCritic is the correct model type
from models.pong_actor_critic import PongActorCritic
from training.rollout import calculate_gae, run_pong_episode # Use the new rollout function
from env.pong import init_env, reset_env, step_env, NUM_ACTIONS, OBSERVATION_SHAPE


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""

    learning_rate: float
    clip_ratio: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    gamma: float
    gae_lambda: float
    update_epochs: int
    seed: int
    batch_size: int # Add batch size for environment vectorization
    # num_steps: int # REMOVED: Not relevant for full-episode rollouts


class PPOTrainer:
    """
    PPO implementation for a single agent (e.g., Atari).
    """

    def __init__(
        self,
        actor_critic: PongActorCritic, # Single ActorCritic model
        config: PPOConfig,
    ):
        """
        Initialize the PPO trainer.

        Args:
            actor_critic: ActorCritic model instance.
            config: PPO configuration parameters.
        """
        self.actor_critic = actor_critic
        self.clip_ratio = config.clip_ratio
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.update_epochs = config.update_epochs
        self.batch_size = config.batch_size # For env vectorization (though Pong wrapper currently supports B=1)
        # self.num_steps = config.num_steps # REMOVED

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )

        self.rng = jax.random.PRNGKey(config.seed)

        # Initialize environment state
        env_rng, self.rng = jax.random.split(self.rng)
        # NOTE: Pong env currently only supports B=1. Adapt if needed.
        self.env_state = init_env(B=1, rng=env_rng)

    def prepare_batch(
        self, trajectory: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Process collected trajectories into training batches.
        Assumes trajectories always contain the terminal state.

        Args:
            trajectory: Collected trajectory data.

        Returns:
            dict: Processed batch data including returns and advantages.
        """
        # GAE Calculation uses the collected values. last_value is not needed.
        rewards = trajectory["rewards"]
        values = trajectory["values"] # Values estimated during rollout
        dones = trajectory["dones"] # Done flags for each step

        advantages = calculate_gae(rewards, values, dones, self.gamma, self.gae_lambda)
        returns = advantages + values # N.B. Values here are V(s_t) from rollout

        # Reshape for training: Flatten T and B dimensions
        # Ensure trajectory doesn't contain 'last_obs' anymore
        batch = {k: v.reshape(-1, *v.shape[2:]) for k, v in trajectory.items() if k != "T"}

        batch["returns"] = returns.reshape(-1)
        batch["advantages"] = advantages.reshape(-1)

        # Normalize advantages
        batch["advantages"] = (batch["advantages"] - jnp.mean(batch["advantages"])) / (
            jnp.std(batch["advantages"]) + 1e-8
        )

        return batch

    def update(
        self,
        params: Any,
        batch: Dict[str, jnp.ndarray],
        opt_state: optax.OptState,
    ) -> Tuple[Any, optax.OptState, Dict[str, jnp.ndarray]]:
        """
        Perform PPO updates for the single agent over multiple epochs.
        (Code is similar to the two-player version, just without distinguishing black/white)
        """

        def loss_fn(params: Any) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            """Calculates the PPO loss and associated metrics."""
            # Use the single agent's actor_critic model
            # Call evaluate_actions via apply, passing the params
            new_log_probs, entropy, values = self.actor_critic.apply(
                {'params': params},
                batch["obs"],
                batch["actions"],
                method=self.actor_critic.evaluate_actions
            )

            ratio = jnp.exp(new_log_probs - batch["log_probs"])
            advantages = batch["advantages"]

            surrogate1 = ratio * advantages
            surrogate2 = (
                jnp.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )
            policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

            value_loss = jnp.mean(jnp.square(values - batch["returns"]))
            entropy_loss = -jnp.mean(entropy)

            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            approx_kl = jnp.mean((ratio - 1) - jnp.log(ratio))
            clip_fraction = jnp.mean(jnp.greater(jnp.abs(ratio - 1.0), self.clip_ratio))

            metrics = {
                "total_loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": -entropy_loss,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            }
            return total_loss, metrics

        #@jax.jit
        def update_epoch(
            carry: Tuple[Any, optax.OptState], _
        ) -> Tuple[Tuple[Any, optax.OptState], Dict[str, jnp.ndarray]]:
            params, opt_state = carry
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_state), metrics

        (new_params, new_opt_state), metrics_history = jax.lax.scan(
            update_epoch, (params, opt_state), None, length=self.update_epochs
        )
        metrics = jax.tree_map(jnp.mean, metrics_history)
        return new_params, new_opt_state, metrics

    def train_step(
        self,
        params: Any,
        optimizer_state: optax.OptState,
    ) -> Tuple[Any, optax.OptState, Dict[str, jnp.ndarray], Any]: # Return updated env_state
        """
        Performs a single training step including rollout and update.

        Args:
            params: Current model parameters.
            optimizer_state: Current optimizer state.

        Returns:
            tuple: (updated_params, updated_opt_state, metrics, updated_env_state, updated_rng)
        """
        # Generate trajectory using the Pong-specific rollout
        rollout_rng, self.rng = jax.random.split(self.rng)
        trajectory, final_env_state, _ = run_pong_episode(
            self.env_state, self.actor_critic, params, rollout_rng
        )

        # Update env_state for the next step
        self.env_state = final_env_state

        # Prepare batch - No longer needs params
        batch = self.prepare_batch(trajectory)

        # Update agent
        new_params, new_opt_state, metrics = self.update(
            params,
            batch,
            optimizer_state,
        )

        # Add episode return/length to metrics if available
        metrics["episode_return"] = trajectory["rewards"].sum()
        metrics["episode_length"] = trajectory["T"]

        return new_params, new_opt_state, metrics, self.rng
