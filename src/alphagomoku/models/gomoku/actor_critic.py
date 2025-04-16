import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
import distrax


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""

    @nn.compact
    def __call__(self, x):
        residual = x

        # First convolution
        y = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        y = nn.LayerNorm()(y)
        y = nn.relu(y)

        # Second convolution
        y = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(y)
        y = nn.LayerNorm()(y)

        # Skip connection and final activation
        return nn.relu(residual + y)


class ActorCritic(nn.Module):
    board_size: int
    channels: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[distrax.Categorical, jnp.ndarray]:
        """
        Forward pass through the network.

        Args:
            x (jnp.ndarray): The board state image with shape (..., board_size, board_size).
                             Allows for batched or unbatched input.

        Returns:
            pi (distrax.Categorical): Policy distribution over actions (flattened board).
            value (jnp.ndarray): Estimated state value with shape (...).
        """
        # Store original shape prefix (e.g., (batch,) or (T, batch))
        prefix_shape = x.shape[:-2]
        board_shape = x.shape[-2:]

        # Reshape input to (total_elements, board_size, board_size) for processing
        x = x.reshape(-1, *board_shape)

        # Add channel dimension if necessary
        if x.shape[-1] != self.channels:
            x = jnp.expand_dims(x, axis=-1)
            if x.shape[-1] != self.channels:
                 raise ValueError(f"Input shape {x.shape} incompatible with channels {self.channels}")


        # Initial convolutional block
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Residual blocks (×6)
        for _ in range(6):
            x = ResidualBlock()(x)

        # Actor Network (Policy Head)
        policy = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        policy = nn.LayerNorm()(policy)
        policy = nn.relu(policy)

        # Final policy output
        policy_logits = nn.Conv(features=1, kernel_size=(1, 1))(policy)
        # Shape: (total_elements, board_size, board_size, 1)
        policy_logits = jnp.squeeze(policy_logits, axis=-1)
        # Shape: (total_elements, board_size, board_size)

        # Flatten logits for Categorical distribution
        flat_policy_logits = policy_logits.reshape(policy_logits.shape[0], -1)
        # Shape: (total_elements, board_size*board_size)

        # Reshape flat logits back to original prefix shape
        final_logits_shape = prefix_shape + (self.board_size * self.board_size,)
        flat_policy_logits = flat_policy_logits.reshape(final_logits_shape)
        # Shape: (..., board_size*board_size)

        pi = distrax.Categorical(logits=flat_policy_logits)

        # Critic Network (Value Head)
        value = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)

        # Global Average Pooling
        value = jnp.mean(value, axis=(1, 2))  # Shape: (total_elements, 32)

        # Fully connected layers
        value = nn.Dense(features=256)(value)
        value = nn.relu(value)
        value = nn.Dense(features=64)(value)
        value = nn.relu(value)
        value = nn.Dense(features=1)(value)
        value = nn.tanh(value)  # Use tanh to constrain between -1 and 1
        value = jnp.squeeze(value, axis=-1)  # Shape: (total_elements,)

        # Reshape value back to original prefix shape
        final_value_shape = prefix_shape
        value = value.reshape(final_value_shape) # Shape: (...)

        return pi, value

    def evaluate_actions(
        self, obs: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Evaluate actions taken, providing log probabilities, entropy, and values.
        Compatible with PPO updates.

        Args:
            obs (jnp.ndarray): Observations (board states) with shape (..., board_size, board_size).
            actions (jnp.ndarray): Actions taken with shape (..., 2) representing (row, col).
                                  These should correspond to the observations.

        Returns:
            tuple: (log_prob, entropy, value)
                   log_prob: Log probability of the actions, shape (...).
                   entropy: Entropy of the policy distribution, shape (...).
                   value: Estimated state value, shape (...).
        """
        # Get policy distribution and value estimate
        pi, value = self(obs) # obs shape (..., board_size, board_size)

        # Convert (row, col) actions to flat indices for distrax
        # actions shape (..., 2)
        flat_actions = actions[..., 0] * self.board_size + actions[..., 1]
        # flat_actions shape (...)

        # Calculate log probability of the taken actions
        log_prob = pi.log_prob(flat_actions) # log_prob shape (...)

        # Calculate entropy of the policy distribution
        entropy = pi.entropy() # entropy shape (...)

        return log_prob, entropy, value 