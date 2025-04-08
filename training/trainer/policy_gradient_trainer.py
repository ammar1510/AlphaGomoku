import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import Dict, Tuple, Any, Optional

from models.actor_critic import ActorCritic


class PolicyGradientTrainer:
    """
    Vanilla policy gradient implementation.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        learning_rate: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        """
        Initialize the policy gradient trainer.

        Args:
            actor_critic: ActorCritic model instance
            learning_rate: Learning rate for optimizer
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            gamma: Discount factor
            seed: Random seed
        """
        self.actor_critic = actor_critic
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma

        # Initialize optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
        )

        # Initialize random key
        self.rng = jax.random.PRNGKey(seed)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, opt_state, trajectory):
        """
        Perform one training update with masked loss computation.

        Args:
            params: ActorCritic model parameters
            opt_state: Optimizer state
            trajectory: Collected trajectory with masks

        Returns:
            tuple: (updated_params, updated_opt_state, loss, aux, grad_norm)
        """
        masks = trajectory["masks"]
        returns = trajectory["rewards"]

        # Normalize returns
        returns_mean = jnp.mean(returns, where=masks)
        returns_std = jnp.std(returns, where=masks) + 1e-8
        normalized_returns = (returns - returns_mean) / returns_std

        def loss_fn(params):
            obs = trajectory["obs"]
            actions = trajectory["actions"]

            board_size = obs.shape[2]  # obs shape is (T, B, board_size, board_size)
            T, B = obs.shape[0], obs.shape[1]

            obs_flat = obs.reshape(-1, board_size, board_size)
            actions_flat = actions.reshape(-1, 2)
            flat_actions = actions_flat[:, 0] * board_size + actions_flat[:, 1]
            masks_flat = masks.reshape(-1)
            returns_flat = normalized_returns.reshape(-1)

            logits, values = self.actor_critic.apply(params, obs_flat)
            values = values.reshape(-1)

            logits_flat = logits.reshape(-1, board_size * board_size)
            log_probs = jax.nn.log_softmax(logits_flat, axis=-1)

            batch_indices = jnp.arange(T * B)
            chosen_log_probs = log_probs[batch_indices, flat_actions]

            probs = jax.nn.softmax(logits_flat, axis=-1)
            entropy = -jnp.sum(probs * log_probs, axis=-1)

            advantages = returns_flat - values

            masked_actor_loss = (
                -chosen_log_probs * jax.lax.stop_gradient(advantages) * masks_flat
            )
            masked_critic_loss = jnp.square(returns_flat - values) * masks_flat
            masked_entropy = entropy * masks_flat

            valid_steps_sum = jnp.sum(masks_flat)
            actor_loss = jnp.sum(masked_actor_loss) / valid_steps_sum
            critic_loss = jnp.sum(masked_critic_loss) / valid_steps_sum
            entropy_loss = -jnp.sum(masked_entropy) / valid_steps_sum

            # Combined loss
            total_loss = (
                actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
            )
            return total_loss, (
                actor_loss,
                critic_loss,
                entropy_loss,
                self.entropy_coef,
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Handle None gradients
        grads = jax.tree_util.tree_map(
            lambda g, p: jnp.zeros_like(p) if g is None else g, grads, params
        )

        grad_norm = optax.global_norm(grads)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, aux, grad_norm
