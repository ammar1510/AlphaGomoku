import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple, Any, Optional

from models.actor_critic import ActorCritic
from training.rollout import calculate_gae, calculate_returns, run_episode


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


class PPOTrainer:
    """
    PPO implementation that uses shared trajectory collection for two players.
    """

    def __init__(
        self,
        black_actor_critic: ActorCritic,
        white_actor_critic: ActorCritic,
        config: PPOConfig,
    ):
        """
        Initialize the PPO trainer.

        Args:
            black_actor_critic: ActorCritic model instance for the black player.
            white_actor_critic: ActorCritic model instance for the white player.
            config: PPO configuration parameters.
        """
        self.black_actor_critic = black_actor_critic
        self.white_actor_critic = white_actor_critic
        self.clip_ratio = config.clip_ratio
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.update_epochs = config.update_epochs

        self.black_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )

        self.white_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )

        self.rng = jax.random.PRNGKey(config.seed)

    def prepare_batch(
        self, trajectory: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Process collected trajectories into training batches.

        Args:
            trajectory: Collected trajectory data for a single player.

        Returns:
            dict: Processed batch data including returns and advantages.
        """
        T, B, board_size, _ = trajectory["obs"].shape
        rewards = trajectory["rewards"]
        values = trajectory["values"]
        dones = trajectory["masks"]
        # yet to add reshaping logic

        advantages = calculate_gae(rewards, values, dones, self.gamma, self.gae_lambda)

        returns = values + advantages

        batch = {**trajectory}
        batch["returns"] = returns
        batch["advantages"] = advantages

        batch["advantages"] = (batch["advantages"] - jnp.mean(batch["advantages"])) / (
            jnp.std(batch["advantages"]) + 1e-8
        )

        return batch

    def update(
        self,
        actor_critic: ActorCritic,
        params: Any,  # Typically flax.core.FrozenDict, but Any for generality
        batch: Dict[str, jnp.ndarray],
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
    ) -> Tuple[Any, optax.OptState, Dict[str, jnp.ndarray]]:
        """
        Perform PPO updates for a single agent over multiple epochs.

        Args:
            actor_critic: The ActorCritic model instance.
            params: Current model parameters.
            batch: Processed batch data for training.
            optimizer: The Optax optimizer.
            opt_state: Current optimizer state.

        Returns:
            tuple: (updated_params, updated_opt_state, metrics averaged over epochs).
        """

        def loss_fn(params: Any) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            """Calculates the PPO loss and associated metrics."""
            new_log_probs, entropy, values = actor_critic.evaluate_actions(
                batch["obs"], batch["actions"]
            )

            # print("Shapes - new_log_probs:", new_log_probs.shape, "batch[log_probs]:", batch["log_probs"].shape)

            ratio = jnp.exp(new_log_probs - batch["log_probs"])

            advantages = batch["advantages"]
            # print("Shapes - ratio:", ratio.shape, "advantages:", advantages.shape)

            surrogate1 = ratio * advantages
            surrogate2 = (
                jnp.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )

            policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

            # Ensure shapes are compatible for subtraction
            # print("Shapes - values:", values.shape, "batch[returns]:", batch["returns"].shape)
            value_loss = jnp.mean(jnp.square(values - batch["returns"]))

            entropy_loss = -jnp.mean(entropy)

            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            # Approximate KL divergence and clip fraction for diagnostics
            approx_kl = jnp.mean((ratio - 1) - jnp.log(ratio))
            clip_fraction = jnp.mean(jnp.greater(jnp.abs(ratio - 1.0), self.clip_ratio))

            metrics = {
                "total_loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": -entropy_loss,  # Report positive entropy
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            }

            return total_loss, metrics

        @jax.jit  # Jit the update step for a single epoch
        def update_epoch(
            carry: Tuple[Any, optax.OptState], _
        ) -> Tuple[Tuple[Any, optax.OptState], Dict[str, jnp.ndarray]]:
            """Performs a single gradient update step."""
            params, opt_state = carry
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, params
            )  # Pass params to optimizer update if needed (e.g., for AdamW)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_state), metrics

        # Run scan over update epochs
        (new_params, new_opt_state), metrics_history = jax.lax.scan(
            update_epoch, (params, opt_state), None, length=self.update_epochs
        )

        # Average metrics over epochs
        metrics = jax.tree_map(jnp.mean, metrics_history)

        return new_params, new_opt_state, metrics

    def train_step(
        self,
        black_params: Any,
        white_params: Any,
        black_optimizer_state: optax.OptState,
        white_optimizer_state: optax.OptState,
        env_state: Any,
    ) -> Tuple[
        Any, optax.OptState, Any, optax.OptState, Dict[str, jnp.ndarray], jnp.ndarray
    ]:
        """
        Performs a single training step including rollout and updates for both players.

        Args:
            black_params: Current parameters for the black player's model.
            white_params: Current parameters for the white player's model.
            black_optimizer_state: Current optimizer state for the black player.
            white_optimizer_state: Current optimizer state for the white player.
            env_state: The current state of the environment.

        Returns:
            tuple: (updated_black_params, updated_black_opt_state,
                    updated_white_params, updated_white_opt_state,
                    combined_metrics, updated_rng_key)
        """
        # Generate trajectories for both players from a single episode
        rollout_rng, self.rng = jax.random.split(self.rng)
        black_trajectory, white_trajectory, _ = run_episode(
            env_state,
            self.black_actor_critic,
            black_params,
            self.white_actor_critic,
            white_params,
            rollout_rng,
        )

        # Prepare batches for each player
        black_player_batch = self.prepare_batch(black_trajectory)
        white_player_batch = self.prepare_batch(white_trajectory)

        # Update black player
        new_black_params, new_black_opt_state, black_metrics = self.update(
            self.black_actor_critic,
            black_params,
            black_player_batch,
            self.black_optimizer,
            black_optimizer_state,
        )

        # Update white player
        new_white_params, new_white_opt_state, white_metrics = self.update(
            self.white_actor_critic,
            white_params,
            white_player_batch,
            self.white_optimizer,
            white_optimizer_state,
        )

        # Consolidate metrics
        combined_metrics = {f"black/{k}": v for k, v in black_metrics.items()}
        combined_metrics.update({f"white/{k}": v for k, v in white_metrics.items()})

        # Include average episode return/length if available from run_episode
        # Example: combined_metrics["episode_return"] = black_trajectory["rewards"].sum() # Or some other logic

        return (
            new_black_params,
            new_black_opt_state,
            new_white_params,
            new_white_opt_state,
            combined_metrics,
            self.rng,
        )
