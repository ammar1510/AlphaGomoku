import jax
import jax.numpy as jnp
import optax
from policy.actor_critic import ActorCritic

BLACK_ID = 1
WHITE_ID = -1

class Trainer:
    def __init__(self, black_policy_model, white_policy_model, env,
                 black_optimizer, white_optimizer, rng):
        """
        Args:
            black_policy_model (nn.Module): policy network for the Black player.
            white_policy_model (nn.Module): policy network for the White player.
            env: The Gomoku environment.
            black_optimizer: optimizer for the Black player.
            white_optimizer: optimizer for the White player.
            rng: A JAX PRNGKey.
        """
        self.black_policy_model = black_policy_model
        self.white_policy_model = white_policy_model
        self.env = env
        self.black_optimizer = black_optimizer
        self.white_optimizer = white_optimizer
        self.rng = rng

    def action_to_coord(self, action: int):
        """
        Converts an integer action into board coordinates.
        """
        board_size = self.env.board_size
        row = action // board_size
        col = action % board_size
        return jnp.array([row, col], dtype=jnp.int32)

    def train(self, num_episodes: int,
              black_params, white_params,
              black_opt_state, white_opt_state):
        """
        Trains the two actor–critic networks (Black and White) over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to play per environment.
            black_params: Initial parameters for the Black network.
            white_params: Initial parameters for the White network.
            black_opt_state: Initial optimizer state for Black.
            white_opt_state: Initial optimizer state for White.

        Returns:
            Updated network parameters and optimizer states.
        """
        num_envs = self.env.num_envs
        episodes_done = [0] * num_envs

        # Reset all environments.
        board, current_player, game_over = self.env.reset()
        # Convert board to a float image with an added channel dimension.
        state = board[..., None].astype(jnp.float32)

        while sum(episodes_done) < num_envs * num_episodes:
            # Split RNG for action selection in each environment.
            self.rng, subrng = jax.random.split(self.rng)
            keys = jax.random.split(subrng, num_envs)

            # Partition indices based on current player.
            black_mask = (current_player == BLACK_ID)
            white_mask = (current_player == WHITE_ID)
            black_indices = jnp.nonzero(black_mask)[0]
            white_indices = jnp.nonzero(white_mask)[0]

            # Initialize actions (one per environment).
            actions = jnp.zeros((num_envs,), dtype=jnp.int32)

            # Select actions for Black.
            if black_indices.shape[0] > 0:
                black_states = state[black_indices]
                black_keys = keys[black_indices]
                # Apply the Black network.
                black_policy_logits, _ = self.black_policy_model.apply(black_params, black_states)
                batch_black = black_policy_logits.shape[0]
                black_flat_logits = black_policy_logits.reshape((batch_black, -1))
                black_probs = jax.nn.softmax(black_flat_logits)
                def sample_action(prob, key):
                    return jax.random.categorical(key, jnp.log(prob))
                black_actions = jax.vmap(sample_action)(black_probs, black_keys)
                actions = actions.at[black_indices].set(black_actions)

            # Select actions for White.
            if white_indices.shape[0] > 0:
                white_states = state[white_indices]
                white_keys = keys[white_indices]
                # Apply the White network.
                white_policy_logits, _ = self.white_policy_model.apply(white_params, white_states)
                batch_white = white_policy_logits.shape[0]
                white_flat_logits = white_policy_logits.reshape((batch_white, -1))
                white_probs = jax.nn.softmax(white_flat_logits)
                def sample_action(prob, key):
                    return jax.random.categorical(key, jnp.log(prob))
                white_actions = jax.vmap(sample_action)(white_probs, white_keys)
                actions = actions.at[white_indices].set(white_actions)

            # Convert integer actions to (row, col) coordinates.
            action_coords = jax.vmap(self.action_to_coord)(actions)

            # Step the environment.
            next_board, next_current_player, next_game_over = self.env.step(action_coords)
            next_state = next_board[..., None].astype(jnp.float32)

            # Define rewards: 1.0 if the move ended the game, else 0.0.
            rewards = jnp.where(next_game_over, 1.0, 0.0)
            dones = next_game_over.astype(jnp.float32)

            # --------------------------
            # Update Black Network
            # --------------------------
            if black_indices.shape[0] > 0:
                black_states = state[black_indices]
                black_actions_taken = actions[black_indices]
                black_rewards = rewards[black_indices]
                # Get value estimates for these states.
                _, black_value = self.black_policy_model.apply(black_params, black_states)
                black_advantage = black_rewards - black_value  # simple one-step advantage

                def loss_fn_black(params):
                    logits, value = self.black_policy_model.apply(params, black_states)
                    batch_b = logits.shape[0]
                    flat_logits = logits.reshape((batch_b, -1))
                    log_probs = jax.nn.log_softmax(flat_logits)
                    # One-hot for chosen actions.
                    actions_one_hot = jax.nn.one_hot(black_actions_taken, flat_logits.shape[1])
                    selected_log_probs = jnp.sum(log_probs * actions_one_hot, axis=1)
                    actor_loss = -jnp.mean(selected_log_probs * black_advantage)
                    critic_loss = jnp.mean((value - black_rewards) ** 2)
                    return actor_loss + critic_loss

                black_loss, black_grads = jax.value_and_grad(loss_fn_black)(black_params)
                black_updates, black_opt_state = self.black_optimizer.update(black_grads, black_opt_state)
                black_params = optax.apply_updates(black_params, black_updates)

            # --------------------------
            # Update White Network
            # --------------------------
            if white_indices.shape[0] > 0:
                white_states = state[white_indices]
                white_actions_taken = actions[white_indices]
                white_rewards = rewards[white_indices]
                _, white_value = self.white_policy_model.apply(white_params, white_states)
                white_advantage = white_rewards - white_value

                def loss_fn_white(params):
                    logits, value = self.white_policy_model.apply(params, white_states)
                    batch_w = logits.shape[0]
                    flat_logits = logits.reshape((batch_w, -1))
                    log_probs = jax.nn.log_softmax(flat_logits)
                    actions_one_hot = jax.nn.one_hot(white_actions_taken, flat_logits.shape[1])
                    selected_log_probs = jnp.sum(log_probs * actions_one_hot, axis=1)
                    actor_loss = -jnp.mean(selected_log_probs * white_advantage)
                    critic_loss = jnp.mean((value - white_rewards) ** 2)
                    return actor_loss + critic_loss

                white_loss, white_grads = jax.value_and_grad(loss_fn_white)(white_params)
                white_updates, white_opt_state = self.white_optimizer.update(white_grads, white_opt_state)
                white_params = optax.apply_updates(white_params, white_updates)

            # Prepare for next iteration.
            board, current_player, game_over = next_board, next_current_player, next_game_over
            state = next_state

            # Reset finished environments.
            game_over_list = jnp.array(game_over).tolist()
            finished_indices = [i for i, done in enumerate(game_over_list) if done]
            if finished_indices:
                for i in finished_indices:
                    episodes_done[i] += 1
                finished_indices_arr = jnp.array(finished_indices, dtype=jnp.int32)
                r_board, r_current_player, r_game_over = self.env.reset(env_indices=finished_indices_arr)
                new_state = r_board[..., None].astype(jnp.float32)
                state = state.at[finished_indices_arr].set(new_state[finished_indices_arr])

        print("Training complete. Episodes per environment:", episodes_done)
        return black_params, white_params, black_opt_state, white_opt_state