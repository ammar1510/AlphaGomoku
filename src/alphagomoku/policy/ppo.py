import jax
import jax.numpy as jnp
import jax.lax as lax
import optax
from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple, Any, Optional, Callable

from alphagomoku.models.gomoku.actor_critic import ActorCritic
from alphagomoku.training.rollout import calculate_gae
from alphagomoku.training.sharding import mesh_rules


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    learning_rate: float = 2.5e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    seed: int = 42


class PPOTrainer:
    """
    Provides core PPO computational steps: GAE calculation and update step.
    Designed to be used within a larger training loop that handles
    trajectory generation, data preparation, and state management.
    """

    @staticmethod
    def prepare_batch_for_update(batch_data: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Reshapes arrays in the batch dictionary from (T, B, ...) to (T * B, ...) using jax.tree_map."""
        
        def _reshape_array(array_leaf: jnp.ndarray) -> jnp.ndarray:
            if not hasattr(array_leaf, 'shape') or len(array_leaf.shape) < 2:
                return array_leaf 
            
            T, B = array_leaf.shape[0], array_leaf.shape[1]
            return lax.with_sharding_constraint(array_leaf.reshape((T * B,) + array_leaf.shape[2:]), mesh_rules("batch"))

        return jax.tree_map(_reshape_array, batch_data)

    @staticmethod
    def compute_gae_targets(
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates Generalized Advantage Estimation (GAE) and returns (targets for value function).
        The returned advantages are normalized (mean=0, std=1) over the batch.

        Args:
            rewards: Sequence of rewards, shape (T, B) or (T,).
            values: Sequence of value estimates V(s_t), including V(s_T), shape (T+1, B) or (T+1,).
            dones: Sequence of done flags, shape (T, B) or (T,).
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.

        Returns:
            tuple: (normalized_advantages, returns)
                   - normalized_advantages: Normalized GAE estimates, shape (T, B) or (T,).
                   - returns: Target values for the value function, shape (T, B) or (T,).
        """
        advantages_raw, returns = calculate_gae(
            rewards, values, dones, gamma, gae_lambda
        )

        # Normalize advantages over the batch
        advantages_mean = advantages_raw.mean()
        advantages_std = (
            advantages_raw.std() + 1e-8
        )  # Add epsilon for numerical stability
        advantages_normalized = (advantages_raw - advantages_mean) / advantages_std

        return advantages_normalized, returns

    @staticmethod
    def update_step(
        rng: jax.random.PRNGKey,
        model: ActorCritic,
        params: optax.Params,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        full_batch: Dict[str, jnp.ndarray],
        config: PPOConfig,
    ) -> Tuple[
        jax.random.PRNGKey, optax.Params, optax.OptState, Dict[str, jnp.ndarray]
    ]:
        """
        Perform PPO updates for a single agent over multiple epochs using mini-batches.

        Args:
            rng: JAX random key.
            model: The ActorCritic model instance.
            params: Current model parameters.
            optimizer: The Optax optimizer.
            opt_state: Current optimizer state.
            full_batch: Preprocessed batch data containing keys like
                         'observations', 'actions', 'logprobs_old', 'advantages', 'returns'.
                         Data should be shaped (N, ...) where N is the total number of steps.
                         Advantages should typically be normalized *before* calling this function.
            config: PPO configuration dataclass.

        Returns:
            tuple: (updated_rng, updated_params, updated_opt_state, metrics averaged over epochs).
        """

        def loss_fn(
            params: optax.Params, minibatch: Dict[str, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            """Calculates the PPO loss and metrics for one mini-batch."""
            obs = minibatch["observations"]
            actions = minibatch["actions"]
            logprobs_old = minibatch["logprobs_old"]
            advantages = minibatch["advantages"]
            returns = minibatch["returns"]
            current_players = minibatch["current_players"]
            valid_mask = minibatch["valid_mask"]

            # --- Get new policy distribution, value, logprobs, and entropy ---
            # Use the evaluate_actions method via model.apply
            logprobs_new, entropy, value_new = model.apply(
                {"params": params},
                obs,
                current_players,
                actions,
                method=model.evaluate_actions,
            )
            # ---

            mask_sum = jnp.sum(valid_mask)

            # --- Value Loss (MSE) - Masked Mean using jnp.mean(where=...) ---
            value_loss_unmasked = (value_new - returns) ** 2
            value_loss = jnp.mean(value_loss_unmasked, where=valid_mask)

            # --- Policy Loss - Masked Mean using jnp.mean(where=...) ---
            log_ratio = logprobs_new - logprobs_old
            ratio = jnp.exp(log_ratio)
            pg_loss1 = advantages * ratio
            pg_loss2 = advantages * jnp.clip(
                ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps
            )
            policy_loss_unmasked = -jnp.minimum(pg_loss1, pg_loss2)
            policy_loss = jnp.mean(policy_loss_unmasked, where=valid_mask)

            # --- Entropy Loss - Masked Mean using jnp.mean(where=...) ---
            entropy_bonus_unmasked = -entropy
            entropy_bonus = jnp.mean(entropy_bonus_unmasked, where=valid_mask)

            total_loss = (
                policy_loss
                + config.vf_coef * value_loss  # Apply vf_coef here
                + config.entropy_coef * entropy_bonus
            )

            # --- Approximate KL Divergence - Masked Mean using jnp.mean(where=...) ---
            approx_kl_unmasked = (ratio - 1.0) - log_ratio
            approx_kl = jnp.mean(approx_kl_unmasked, where=valid_mask)

            # --- Clip Fraction - Masked Mean using jnp.mean(where=...) ---
            # Need to cast boolean clip fraction to float for mean
            clip_fraction_unmasked = jnp.abs(ratio - 1.0) > config.clip_eps
            clip_fraction = jnp.mean(
                clip_fraction_unmasked.astype(jnp.float32), where=valid_mask
            )

            metrics = {
                "total_loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": -entropy_bonus,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "mask_sum_fraction": mask_sum / valid_mask.size,  # Re-added metric
            }
            return total_loss, metrics

        @partial(jax.jit, static_argnames=["optimizer", "model"])
        def _update_minibatch(
            params: optax.Params,
            opt_state: optax.OptState,
            optimizer: optax.GradientTransformation,
            model: ActorCritic,
            minibatch: Dict[str, jnp.ndarray], # (N, ...)
        ) -> Tuple[optax.Params, optax.OptState, Dict[str, jnp.ndarray]]:
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, metrics), grads = grad_fn(params, minibatch)

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            metrics["step_loss"] = loss
            return new_params, new_opt_state, metrics

        metrics_history = []
        total_data_points = full_batch["advantages"].size


        batch_size = total_data_points // config.num_minibatches
        effective_num_minibatches = config.num_minibatches

        indices = jnp.arange(total_data_points)

        for epoch in range(config.update_epochs):
            rng, shuffle_rng = jax.random.split(rng)
            shuffled_indices = jax.random.permutation(shuffle_rng, indices)

            for i in range(effective_num_minibatches):
                start_idx = i * batch_size
                end_idx = (
                    start_idx + batch_size
                    if i < effective_num_minibatches - 1
                    else total_data_points
                )

                mb_indices = shuffled_indices[start_idx:end_idx]

                mini_batch = jax.tree.map(lambda x: x[mb_indices], full_batch)

                params, opt_state, step_metrics = _update_minibatch(
                    params, opt_state, optimizer, model, mini_batch
                )
                metrics_history.append(step_metrics)

        if not metrics_history:
            final_metrics = {}
        else:
            final_metrics = jax.tree.map(
                lambda *xs: jnp.mean(jnp.stack(xs)), *metrics_history
            )

        return rng, params, opt_state, final_metrics
