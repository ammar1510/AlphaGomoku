import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


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
    def __call__(self, x):
        """
        Args:
            x (jnp.ndarray): The board state image with shape (batch, board_size, board_size)

        Returns:
            policy_logits (jnp.ndarray): Logits over actions with shape (batch, board_size, board_size).
            value (jnp.ndarray): Estimated state value with shape (batch,).
        """
        # Add channel dimension since it's missing
        x = jnp.expand_dims(x, axis=-1)

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
        policy_logits = jnp.squeeze(
            policy_logits, axis=-1
        )  # Shape: (batch, board_size, board_size)

        # Legal move masking should be applied during training/inference outside this module
        # as we don't have the original board state here to create the mask

        # Critic Network (Value Head)
        value = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)

        # Global Average Pooling (equivalent to avg_pool over the entire spatial dimensions)
        value = jnp.mean(value, axis=(1, 2))  # Shape: (batch, 32)

        # Fully connected layers
        value = nn.Dense(features=256)(value)
        value = nn.relu(value)
        value = nn.Dense(features=64)(value)
        value = nn.relu(value)
        value = nn.Dense(features=1)(value)
        value = nn.tanh(value)  # Use tanh to constrain between -1 and 1
        value = jnp.squeeze(value, axis=-1)  # Shape: (batch,)

        return policy_logits, value

    def sample_action(self, logits, rng):
        """
        Samples an action from the given logits and returns a tuple (row, col).

        Args:
            logits (jnp.ndarray): A tensor of shape (batch, board_size, board_size)
                                  representing the unnormalized log-probabilities.
            rng: A JAX PRNGKey for randomness.

        Returns:
            jnp.ndarray: Selected moves as an array with shape (batch, 2).
        """
        batch_size, board_size, _ = logits.shape
        flat_logits = logits.reshape((batch_size, board_size * board_size))
        flat_actions = jax.random.categorical(rng, flat_logits, axis=-1)
        rows = flat_actions // board_size
        cols = flat_actions % board_size
        return jnp.stack([rows, cols], axis=1)

    def mask_invalid_actions(self, logits, board):
        """
        Masks invalid moves by setting their logits to a large negative value.

        Args:
            logits (jnp.ndarray): Policy logits with shape (batch, board_size, board_size)
            board (jnp.ndarray): Board state with same shape, where 0 indicates empty spaces
                                 that are valid moves

        Returns:
            jnp.ndarray: Masked logits with same shape as input
        """
        # Create mask where True represents empty spaces (valid moves)
        valid_moves = board == 0

        # Apply mask - set invalid moves to large negative value
        masked_logits = jnp.where(valid_moves, logits, -jnp.inf)

        return masked_logits

    def get_log_prob(self, logits, actions):
        """
        Calculate log probabilities for specific actions.

        Args:
            logits (jnp.ndarray): Policy logits with shape (batch, board_size, board_size)
            actions (jnp.ndarray): Actions with shape (batch, 2) representing (row, col)

        Returns:
            jnp.ndarray: Log probabilities for the given actions, shape (batch,)
        """
        batch_size, board_size, _ = logits.shape
        flat_logits = logits.reshape((batch_size, board_size * board_size))

        # Convert (row, col) actions to flat indices
        flat_actions = actions[:, 0] * board_size + actions[:, 1]

        action_logits = flat_logits[jnp.arange(batch_size), flat_actions]

        log_probs = jax.nn.log_softmax(action_logits, axis=-1)

        return log_probs

    def evaluate_actions(
        self, states, actions
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Evaluate states and actions to get log probabilities, entropy, and values.

        Args:
            states (jnp.ndarray): States with shape (T, batch, board_size, board_size)
            actions (jnp.ndarray): Actions with shape (T, batch, 2)

        Returns:
            tuple: (log_probs, entropy, values)
        """
        # Forward pass through the network
        logits, values = self(states)

        # Calculate log probabilities
        T, batch_size, board_size, _ = logits.shape
        flat_logits = logits.reshape((T, batch_size, board_size * board_size))
        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)

        # Convert actions to flat indices and get log probs
        flat_actions = actions[:, :, 0] * board_size + actions[:, :, 1]
        batch_indices = jnp.arange(batch_size)
        action_log_probs = log_probs[batch_indices, flat_actions]

        # Calculate entropy
        probs = jax.nn.softmax(flat_logits, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        return action_log_probs, entropy, values
