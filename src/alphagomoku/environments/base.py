import abc
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Type, Optional, TypeVar

# Placeholder TypeVar for the environment-specific state (e.g., GomokuState)
# Subclasses will define their own specific state type, likely a NamedTuple or dataclass.
EnvState = TypeVar("EnvState")


class Env(abc.ABC):
    """
    Abstract Base Class for JAX-based environment logic containers.

    This class holds static configuration (e.g., board size, win length)
    and defines the interface for pure functions that operate on an immutable
    environment state object (`EnvState`).

    Subclasses should:
    1. Define a specific `EnvState` structure (e.g., a NamedTuple or dataclass
       registered as a JAX PyTree) to hold dynamic JAX arrays (board, player, etc.).
    2. Implement the abstract methods below as pure functions (often staticmethods
       or methods using `self` only for static configuration access).
    """

    def __init__(self, B: int, **env_specific_config):
        """
        Initializes the environment logic container with static configuration.

        Args:
            B: Batch size.
            **env_specific_config: Environment-specific static configuration arguments
                                     (e.g., board_size, win_length).
        """
        self.B = B  # Store batch size, often needed by logic functions
        # Subclass implementation should store necessary static config from
        # env_specific_config as attributes of `self`.
        super().__init__()

    @abc.abstractmethod
    def step(
        self, state: EnvState, actions: jnp.ndarray
    ) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Applies actions to the current state to compute the next state and outputs. Pure function.

        Args:
            state: The current environment state (an instance of the subclass-defined EnvState).
            actions: A JAX array containing the actions for each env in the batch.
                     Shape depends on the specific environment's action space.

        Returns:
            A tuple (new_state, observations, rewards, dones, info):
                - new_state: The next environment state (an instance of EnvState).
                - observations: The next observations. Shape depends on the obs space.
                - rewards: The rewards received. Shape (B,).
                - dones: Boolean flags indicating episode termination. Shape (B,).
                - info: Auxiliary dictionary (can be empty).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
        self
    ) -> Tuple[EnvState, jnp.ndarray, Dict[str, Any]]:
        """
        Creates an initial environment state. Pure function.

        Returns:
            A tuple (initial_state, initial_observations, info):
                - initial_state: The initial EnvState.
                - initial_observations: Observations after reset. Shape depends on obs space.
                - info: Auxiliary dictionary.
        """
        raise NotImplementedError

    # Note: This might be better placed in the training loop, but kept here for now.
    # It should only depend on static configuration.
    @abc.abstractmethod
    def initialize_trajectory_buffers(self, max_steps: int) -> Tuple[jnp.ndarray, ...]:
        """
        Creates and returns pre-allocated JAX arrays for storing trajectory data,
        based on the environment's static configuration (shapes, dtypes).

        Args:
            max_steps: The maximum length of the trajectories to buffer.

        Returns:
            A tuple containing JAX arrays (e.g., for observations, actions,
            rewards, masks, logprobs) dimensioned with `max_steps` and `self.B`.
            Exact tuple contents and shapes/dtypes depend on the subclass.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def observation_shape(self) -> tuple:
        """Returns the shape tuple of a single observation (excluding batch dim)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_shape(self) -> tuple:
        """Returns the shape tuple of a single action (excluding batch dim)."""
        raise NotImplementedError

    # --- Optional Methods ---

    @abc.abstractmethod
    def get_action_mask(self, state: EnvState) -> jnp.ndarray:
        """
        Returns a boolean mask of valid actions for the given state. Pure function.

        Args:
            state: The current environment state (an instance of the subclass-defined EnvState).

        Returns:
            A boolean JAX array mask (True=valid, False=invalid).
            The shape must be compatible with the policy network's output logits
            (e.g., shape (B, num_actions) or (B, board_height, board_width)).
        """
        raise NotImplementedError
