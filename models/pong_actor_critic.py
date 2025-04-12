import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Tuple
import distrax # Assuming distrax is available for action distributions

class PongActorCritic(nn.Module):
    """
    Simple MLP Actor-Critic network for Pong RAM observations.
    """
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Categorical, jnp.ndarray]:
        """
        Forward pass through the network.

        Args:
            x: Input observation (Pong RAM state), shape (B, 128).

        Returns:
            pi: Action distribution (distrax.Categorical).
            value: Estimated state value, shape (B,).
        """
        # Normalize RAM observations (optional but often helpful)
        # RAM values are 0-255
        x = x.astype(jnp.float32) / 255.0

        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        # Shared layers
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation_fn(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation_fn(actor_mean)

        # Actor head
        pi_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=pi_logits)

        # Critic head
        critic = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation_fn(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation_fn(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(value, axis=-1) # Return value squeezed

    def evaluate_actions(self, obs: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Evaluate actions taken, providing log probabilities, entropy, and values.
        Used during PPO updates.
        """
        pi, value = self(obs)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()
        return log_prob, entropy, value 