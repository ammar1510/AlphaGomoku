import jax.numpy as jnp
from flax import linen as nn

class ActorCritic(nn.Module):
    board_size: int  # e.g. 15
    channels: int = 1  # Number of input channels (for a plain board image)

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x (jnp.ndarray): The board state image with shape
                             (batch, board_size, board_size, channels).
        
        Returns:
            policy_logits (jnp.ndarray): Logits over actions with shape (batch, board_size*board_size).
            value (jnp.ndarray): Estimated state value with shape (batch,).
        """
        # Shared Convolutional Backbone
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        # Actor Head:
        # Use a 1x1 convolution to reduce the features to a single logit per board cell.
        actor = nn.Conv(features=1, kernel_size=(1, 1), padding="SAME")(x)
        # Remove the channel dimension: shape becomes (batch, board_size, board_size)
        policy_logits = jnp.squeeze(actor, axis=-1) # shape: (batch, board_size, board_size)

        # Critic Head:
        # Use global average pooling to collapse the spatial dimensions.
        critic = nn.avg_pool(x, window_shape=(self.board_size, self.board_size), strides=(1, 1), padding="VALID")
        # Now shape: (batch, 1, 1, features), squeeze to (batch, features)
        critic = jnp.squeeze(critic, axis=(1, 2))
        # Further process with an MLP for a robust value estimate.
        critic = nn.Dense(features=256)(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(features=1)(critic)
        value = jnp.squeeze(critic, axis=-1)  # shape (batch,)

        return policy_logits, value 