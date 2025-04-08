import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import Dict, Tuple, Any, Optional

from models.actor_critic import ActorCritic
from training.rollout import collect_selfplay_trajectories
from training.trainer.policy_gradient_trainer import PolicyGradientTrainer


class SelfPlayTrainer:
    """
    Self-play trainer for Gomoku.

    Uses separate models for black and white players with the option
    to randomly swap between checkpoint versions.
    """

    def __init__(
        self,
        black_actor_critic: ActorCritic,
        white_actor_critic: ActorCritic,
        learning_rate: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        """
        Initialize the self-play trainer.

        Args:
            black_actor_critic: ActorCritic model for black player
            white_actor_critic: ActorCritic model for white player
            learning_rate: Learning rate for optimizer
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            gamma: Discount factor
            seed: Random seed
        """
        self.black_actor_critic = black_actor_critic
        self.white_actor_critic = white_actor_critic
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma

        # Initialize optimizers
        self.black_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
        )

        self.white_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
        )

        # Initialize random key
        self.rng = jax.random.PRNGKey(seed)

    def train_step(
        self,
        black_params,
        black_opt_state,
        white_params,
        white_opt_state,
        env_state,
        entropy_coef=None,
    ):
        """
        Perform one training step with self-play.

        Args:
            black_params: Black player model parameters
            black_opt_state: Black player optimizer state
            white_params: White player model parameters
            white_opt_state: White player optimizer state
            env_state: Environment state
            entropy_coef: Optional override for entropy coefficient

        Returns:
            tuple: (black_params, black_opt_state, white_params, white_opt_state,
                   black_metrics, white_metrics, final_env_state)
        """
        # Use entropy coefficient override if provided
        entropy_coef = entropy_coef if entropy_coef is not None else self.entropy_coef

        # Collect trajectories with self-play
        self.rng, subkey = jax.random.split(self.rng)
        black_traj, white_traj, self.rng = collect_selfplay_trajectories(
            env_state,
            self.black_actor_critic,
            black_params,
            self.white_actor_critic,
            white_params,
            self.gamma,
            subkey,
        )

        # Create policy gradient trainers for each player
        black_trainer = PolicyGradientTrainer(
            self.black_actor_critic, entropy_coef=entropy_coef
        )

        white_trainer = PolicyGradientTrainer(
            self.white_actor_critic, entropy_coef=entropy_coef
        )

        # Update black player
        black_params, black_opt_state, black_loss, black_aux, black_grad_norm = (
            black_trainer.train_step(black_params, black_opt_state, black_traj)
        )

        # Update white player
        white_params, white_opt_state, white_loss, white_aux, white_grad_norm = (
            white_trainer.train_step(white_params, white_opt_state, white_traj)
        )

        # Create metrics dictionaries
        black_metrics = {
            "loss": black_loss,
            "actor_loss": black_aux[0],
            "critic_loss": black_aux[1],
            "entropy_loss": black_aux[2],
            "entropy_coef": black_aux[3],
            "grad_norm": black_grad_norm,
        }

        white_metrics = {
            "loss": white_loss,
            "actor_loss": white_aux[0],
            "critic_loss": white_aux[1],
            "entropy_loss": white_aux[2],
            "entropy_coef": white_aux[3],
            "grad_norm": white_grad_norm,
        }

        return (
            black_params,
            black_opt_state,
            white_params,
            white_opt_state,
            black_metrics,
            white_metrics,
        )
