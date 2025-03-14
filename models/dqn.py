from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


class DuelingDQN(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Feature extraction
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Separate streams
        value_stream = nn.Dense(1)(x)
        advantage_stream = nn.Dense(self.action_dim)(x)

        # Combine streams using dueling architecture
        q_values = value_stream + (
            advantage_stream - jnp.mean(advantage_stream, axis=-1, keepdims=True)
        )
        return q_values


class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Initialize network and optimizer
        self.model = DuelingDQN(action_dim)
        self.optimizer = optax.adam(learning_rate)

        # Initialize parameters
        self.params = self.model.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
        self.opt_state = self.optimizer.init(self.params)

        # Manage a PRNG key in the agent's state
        self.key = jax.random.PRNGKey(0)

    def get_action(self, state, epsilon):
        # Split key to generate new subkeys
        self.key, subkey = jax.random.split(self.key)
        if jax.random.uniform(subkey) < epsilon:
            self.key, subkey = jax.random.split(self.key)
            return jax.random.randint(subkey, (), 0, self.action_dim)
        else:
            q_values = self.model.apply(self.params, state)
            return jnp.argmax(q_values)

    def update(self, states, actions, rewards, next_states, dones):
        def loss_fn(params):
            # Current Q values
            q_values = self.model.apply(params, states)
            current_q = jnp.take_along_axis(
                q_values, actions[:, None], axis=1
            ).squeeze()

            # Target Q values
            next_q_values = self.model.apply(params, next_states)
            max_next_q = jnp.max(next_q_values, axis=1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

            # MSE loss
            return jnp.mean((current_q - target_q) ** 2)

        # Compute gradients and update parameters
        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)

        return loss
