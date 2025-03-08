import jax.numpy as jnp
from flax import linen as nn
import jax

class ActorCritic(nn.Module):
    board_size: int
    channels: int = 1

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x (jnp.ndarray): The board state image with shape either:
                             (batch, board_size, board_size, channels)
                             or (batch, board_size, board_size) if the channel dim is missing.

        Returns:
            policy_logits (jnp.ndarray): Logits over actions with shape (batch, board_size, board_size).
            value (jnp.ndarray): Estimated state value with shape (batch,).
        """
        # If the input is missing the channel dimension (e.g. shape == (batch, board_size, board_size)),
        # add it.
        if x.ndim == 3:
            x = x[..., None]
        elif x.ndim != 4:
            raise ValueError(f"Expected input to have 3 or 4 dimensions, got shape {x.shape}")

        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        residual = x
        y = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        y = nn.relu(y)
        x = residual + y  

        residual = x
        y = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        y = nn.relu(y)
        x = residual + y  

        actor = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(x)
        policy_logits = jnp.squeeze(actor, axis=-1) 

        critic = nn.avg_pool(
            x, window_shape=(self.board_size, self.board_size), strides=(1, 1), padding="VALID")
        critic = jnp.squeeze(critic, axis=(1, 2))
        critic = nn.Dense(features=256)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(features=1)(critic)
        value = jnp.squeeze(critic, axis=-1)

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
