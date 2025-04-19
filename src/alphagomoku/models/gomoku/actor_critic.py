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
    channels: int = 2  # Now expects 2 channels: board state + player turn

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, current_players: jnp.ndarray
    ) -> Tuple[distrax.Categorical, jnp.ndarray]:
        """
        Forward pass through the network.

        Args:
            x (jnp.ndarray): The board state image with shape (..., board_size, board_size).
                             Allows for batched or unbatched input.
            current_players (jnp.ndarray): Scalar or array indicating the current player(s)
                                          (-1 or 1) with shape (...). Must be broadcastable
                                          to x's prefix shape.

        Returns:
            pi (distrax.Categorical): Policy distribution over actions (..., flattened board).
            value (jnp.ndarray): Estimated state value with shape (...).
        """
        prefix_shape = x.shape[:-2]  # e.g., (batch,) or (T, batch) or ()
        board_shape = x.shape[-2:]  # (board_size, board_size)

        # Add channel dimension for board state
        # Shape: (..., board_size, board_size, 1)
        x_proc = jnp.expand_dims(x, axis=-1)

        # Create player channel
        # Ensure current_players has the correct prefix shape
        player_array = jnp.broadcast_to(current_players, prefix_shape)
        # Reshape player_array to (..., 1, 1, 1) for broadcasting to spatial dims
        player_array_reshaped = player_array.reshape(
            prefix_shape + (1,) * (len(board_shape) + 1)
        )
        # Create the channel plane: (..., board_size, board_size, 1)
        player_channel = jnp.ones_like(x_proc) * player_array_reshaped

        # Concatenate board state and player channel
        # Shape: (..., board_size, board_size, 2)
        x_combined = jnp.concatenate([x_proc, player_channel], axis=-1)

        # --- Network Layers --- Apply layers directly, preserving leading dimensions

        # Initial convolutional block
        # Input: (..., H, W, C_in=2), Output: (..., H, W, 64)
        net = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x_combined)
        net = nn.LayerNorm()(net)  # Normalizes over the last axis (features)
        net = nn.relu(net)

        # Residual blocks (×6)
        # Input/Output: (..., H, W, 64)
        for _ in range(6):
            net = ResidualBlock()(net)

        # --- Actor Head ---
        # Input: (..., H, W, 64), Output: (..., H, W, 32)
        policy = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(net)
        policy = nn.LayerNorm()(policy)
        policy = nn.relu(policy)

        # Final policy output
        # Input: (..., H, W, 32), Output: (..., H, W, 1)
        policy_logits = nn.Conv(features=1, kernel_size=(1, 1))(policy)
        # Squeeze the channel dim: Output: (..., H, W)
        policy_logits = jnp.squeeze(policy_logits, axis=-1)

        # Flatten spatial dimensions for Categorical distribution
        # Input: (..., H, W), Output: (..., H*W)
        flat_policy_logits = policy_logits.reshape(prefix_shape + (-1,))

        pi = distrax.Categorical(logits=flat_policy_logits)

        # --- Critic Head ---
        # Input: (..., H, W, 64), Output: (..., H, W, 32)
        value = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(net)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)

        # Global Average Pooling over spatial dimensions (H, W)
        # Input: (..., H, W, 32), Output: (..., 32)
        value = jnp.mean(
            value, axis=(-3, -2)
        )  # Axes H, W are second and third from last

        # Fully connected layers
        # Input: (..., 32), Output: (..., 256)
        value = nn.Dense(features=256)(value)
        value = nn.relu(value)
        # Input: (..., 256), Output: (..., 64)
        value = nn.Dense(features=64)(value)
        value = nn.relu(value)
        # Input: (..., 64), Output: (..., 1)
        value = nn.Dense(features=1)(value)
        # Output: (..., 1)
        value = nn.tanh(value)  # Use tanh to constrain between -1 and 1
        # Squeeze the last dim: Output: (...)
        value = jnp.squeeze(value, axis=-1)

        return pi, value

    def evaluate_actions(
        self, obs: jnp.ndarray, current_players: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Evaluate actions taken, providing log probabilities, entropy, and values.
        Compatible with PPO updates.

        Args:
            obs (jnp.ndarray): Observations (board states) with shape (..., board_size, board_size).
            current_players (jnp.ndarray): Player (-1 or 1) corresponding to each observation, shape (...).
            actions (jnp.ndarray): Actions taken with shape (..., 2) representing (row, col).
                                  These should correspond to the observations.

        Returns:
            tuple: (log_prob, entropy, value)
                   log_prob: Log probability of the actions, shape (...).
                   entropy: Entropy of the policy distribution, shape (...).
                   value: Estimated state value, shape (...).
        """
        # Get policy distribution and value estimate
        pi, value = self(obs, current_players)  # Pass renamed argument

        # Convert (row, col) actions to flat indices for distrax
        # actions shape (..., 2)
        flat_actions = actions[..., 0] * self.board_size + actions[..., 1]
        # flat_actions shape (...)

        # Calculate log probability of the taken actions
        log_prob = pi.log_prob(flat_actions)  # log_prob shape (...)

        # Calculate entropy of the policy distribution
        entropy = pi.entropy()  # entropy shape (...)

        return log_prob, entropy, value
