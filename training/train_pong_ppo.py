import jax
import jax.numpy as jnp
import optax
import tqdm # For progress bar
import logging # For logging
import time
import os # For path manipulation
from dataclasses import dataclass, field
import hydra
from omegaconf import DictConfig, OmegaConf
from flax.serialization import msgpack_serialize, msgpack_restore # Import for saving/loading

from env.pong import init_env, NUM_ACTIONS, OBSERVATION_SHAPE
from models.pong_actor_critic import PongActorCritic
from training.trainer.ppo_trainer import PPOTrainer, PPOConfig
# Assume wandb logging is desired
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default PPO configuration using OmegaConf for structure and potential overrides
@dataclass
class TrainConfig:
    # PPO Hyperparameters (matching PPOConfig)
    learning_rate: float = 2.5e-4
    clip_ratio: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    seed: int = 42
    batch_size: int = 1 # Pong env currently supports B=1
    # num_steps: int = 128 # REMOVED: Not used as rollout runs full episodes

    # Training settings
    total_timesteps: int = 1_000_000
    log_interval: int = 10 # Log metrics every N training steps
    save_interval: int = 100 # Save model checkpoint every N updates
    model_save_path: str = "pong_ppo_model.msgpack" # Base path for saving models
    use_wandb: bool = True

    # Model settings
    model_activation: str = "tanh"

    # Create PPOConfig from TrainConfig
    def get_ppo_config(self) -> PPOConfig:
        return PPOConfig(
            learning_rate=self.learning_rate,
            clip_ratio=self.clip_ratio,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            update_epochs=self.update_epochs,
            seed=self.seed,
            batch_size=self.batch_size,
            # num_steps=self.num_steps, # REMOVED
        )

# Using Hydra for configuration management
@hydra.main(version_base=None, config_path="../cfg", config_name="pong_ppo_config") # Corrected path
def main(cfg: DictConfig):
    # Use original DictConfig (cfg) for direct access where possible
    logger.info(f"Starting Pong PPO training with config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Assert CPU Backend --- #
    default_backend = jax.default_backend()
    assert default_backend == 'cpu', f"Expected JAX backend to be 'cpu', but found '{default_backend}'. Pong environment interactions may not be suitable for other backends."
    logger.info(f"Using JAX backend: {default_backend}")
    # --- End Assertion --- #

    # Convert DictConfig to a standard Python dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # Instantiate TrainConfig dataclass from the dictionary
    try:
        train_cfg = TrainConfig(**cfg_dict)
    except TypeError as e:
        logger.error(f"Failed to instantiate TrainConfig from config: {e}")
        logger.error(f"Config Dictionary: {cfg_dict}")
        raise e

    # Seeding - Use train_cfg now as it's a proper instance
    key = jax.random.PRNGKey(train_cfg.seed)
    model_key, trainer_key = jax.random.split(key)

    # Initialize Logging (WandB) - Use train_cfg
    if train_cfg.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="alphagomoku-pong-ppo", # Example project name
            config=cfg_dict, # Log the original dict representation
            sync_tensorboard=False,
            monitor_gym=False, # We are using our own env wrapper
            save_code=True,
        )
        logger.info("Weights & Biases initialized.")
    elif train_cfg.use_wandb:
        logger.warning("WandB requested but not installed. Skipping WandB logging.")
        train_cfg.use_wandb = False # Modify the dataclass instance if needed

    # Initialize Model - Use train_cfg
    actor_critic = PongActorCritic(action_dim=NUM_ACTIONS, activation=train_cfg.model_activation)
    # Dummy observation for initialization
    dummy_obs = jnp.zeros((1,) + OBSERVATION_SHAPE, dtype=jnp.uint8)
    params = actor_critic.init(model_key, dummy_obs)["params"]
    logger.info("Pong Actor-Critic model initialized.")

    # Initialize Trainer - Needs train_cfg for get_ppo_config
    ppo_config = train_cfg.get_ppo_config()
    trainer = PPOTrainer(actor_critic, ppo_config)
    opt_state = trainer.optimizer.init(params)
    logger.info("PPO Trainer initialized.")

    # Training Loop - Loop based on global_step
    logger.info(f"Starting training loop for {train_cfg.total_timesteps} timesteps...")
    start_time = time.time()
    global_step = 0 # Track total environment steps
    update_count = 0 # Track number of updates (episodes)

    # The trainer holds the env_state internally now
    # train_state needs params, opt_state, rng
    train_state = {
        "params": params,
        "opt_state": opt_state,
        "rng": trainer_key,
        # Env state is managed inside trainer
    }

    # Use TQDM for progress bar based on total_timesteps
    pbar = tqdm.tqdm(total=train_cfg.total_timesteps, desc="Training Progress")
    last_global_step = 0

    while global_step < train_cfg.total_timesteps:
        update_count += 1
        # Perform one training step (rollout + update)
        # Call train_step directly without JIT
        new_params, new_opt_state, metrics, new_rng = trainer.train_step(
            train_state["params"],
            train_state["opt_state"],
            # Pass rng from train_state, trainer updates its internal rng too
        )

        # Update train state
        train_state["params"] = new_params
        train_state["opt_state"] = new_opt_state
        train_state["rng"] = new_rng # Trainer now returns the updated key

        current_step_count = metrics["episode_length"]
        previous_global_step = global_step
        global_step += current_step_count.item() # .item() to get scalar from JAX array

        # Update progress bar
        pbar.update(global_step - last_global_step)
        last_global_step = global_step
        pbar.set_postfix({"Return": f"{metrics['episode_return'].item():.2f}"})

        # Logging - Use train_cfg
        # Log based on update_count or time interval if desired, here using log_interval based on updates
        if update_count % train_cfg.log_interval == 0:
            end_time = time.time()
            steps_per_second = (global_step - previous_global_step) / (end_time - start_time) # SPS for last interval
            start_time = end_time # Reset timer for next interval

            log_data = {
                "update": update_count,
                "global_step": global_step,
                "sps": steps_per_second,
                **{k: v.item() for k, v in metrics.items()} # Convert JAX arrays to scalars
            }
            logger.info(f"Update: {update_count}, Global Step: {global_step}, SPS: {steps_per_second:.2f}, Return: {metrics['episode_return']:.2f}")
            if train_cfg.use_wandb and WANDB_AVAILABLE:
                wandb.log(log_data, step=global_step)

        # --- Model Saving Logic --- Use train_cfg
        # Save based on update_count
        if update_count % train_cfg.save_interval == 0:
            # Create a directory for checkpoints if it doesn't exist
            checkpoint_dir = os.path.dirname(train_cfg.model_save_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
                logger.info(f"Created checkpoint directory: {checkpoint_dir}")

            # Construct checkpoint path (use update_count)
            base, ext = os.path.splitext(train_cfg.model_save_path)
            checkpoint_path = f"{base}_update_{update_count}{ext}"

            # Save parameters
            try:
                with open(checkpoint_path, "wb") as f:
                    f.write(msgpack_serialize(train_state["params"]))
                logger.info(f"Saved model checkpoint at update {update_count} to {checkpoint_path}")
            except Exception as e:
                 logger.error(f"Failed to save checkpoint at update {update_count}: {e}")
        # --- End Model Saving Logic ---

    pbar.close() # Close the progress bar
    logger.info("Training finished.")

    # Save final model - Use train_cfg
    # Ensure directory exists
    final_save_dir = os.path.dirname(train_cfg.model_save_path)
    if final_save_dir and not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)

    try:
        with open(train_cfg.model_save_path, "wb") as f:
            f.write(msgpack_serialize(train_state["params"]))
        logger.info(f"Final model parameters saved to {train_cfg.model_save_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    if train_cfg.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main() 