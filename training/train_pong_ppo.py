import jax
import jax.numpy as jnp
import optax
import tqdm 
import logging 
import time
import os 
from dataclasses import dataclass, field
import hydra
from omegaconf import DictConfig, OmegaConf
from flax.training import train_state as flax_train_state # For Orbax compatibility
import orbax.checkpoint as ocp # Orbax import
import wandb 

from env.pong import init_env, NUM_ACTIONS, OBSERVATION_SHAPE
from models.pong_actor_critic import PongActorCritic
from training.trainer.ppo_trainer import PPOTrainer, PPOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainState(flax_train_state.TrainState):
    # Inherits apply_fn, params, tx, opt_state
    # Add any other state components you want to checkpoint
    rng: jax.Array
    global_step: int
    update_count: int


@dataclass
class TrainConfig:
    # PPO Hyperparameters
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

    total_timesteps: int = 1_000_000
    log_interval: int = 10
    save_interval_updates: int = 100 
    checkpoint_dir: str = "pong_ppo_checkpoints" 
    model_activation: str = "tanh"

    # Orbax settings
    orbax_max_to_keep: int = 3 # Keep latest 3 checkpoints



@hydra.main(version_base=None, config_path="../cfg", config_name="pong_ppo_config") # Corrected path
def main(cfg: DictConfig):
    logger.info(f"Starting Pong PPO training with config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Assert CPU Backend --- #
    default_backend = jax.default_backend()
    assert default_backend == 'cpu', f"Expected JAX backend to be 'cpu', but found '{default_backend}'. Pong environment interactions may not be suitable for other backends."
    logger.info(f"Using JAX backend: {default_backend}")

    # --- Config and RNG Setup ---
    cfg_obj = OmegaConf.to_object(cfg)
    train_cfg = TrainConfig(**cfg_obj) # Instantiate the dataclass
    key = jax.random.PRNGKey(train_cfg.seed)
    model_key, trainer_key, state_rng_key = jax.random.split(key, 3)

    # --- Wandb Initialization ---
    wandb.init(
        project="alphagomoku-pong-ppo",
        config=cfg_obj, # Log the original DictConfig
        sync_tensorboard=False,
        monitor_gym=False,
        save_code=True,
    )
    logger.info("Weights & Biases initialized.")


    # --- Model Initialization ---
    actor_critic = PongActorCritic(action_dim=NUM_ACTIONS, activation=train_cfg.model_activation)
    dummy_obs = jnp.zeros((1,) + OBSERVATION_SHAPE, dtype=jnp.uint8)
    initial_variables = actor_critic.init(model_key, dummy_obs)
    logger.info("Pong Actor-Critic model initialized.")


    # --- Optimizer Initialization ---
    # Create PPOConfig from TrainConfig for the trainer
    ppo_config = PPOConfig(
        learning_rate=train_cfg.learning_rate,
        clip_ratio=train_cfg.clip_ratio,
        value_coef=train_cfg.value_coef,
        entropy_coef=train_cfg.entropy_coef,
        max_grad_norm=train_cfg.max_grad_norm,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        update_epochs=train_cfg.update_epochs,
        seed=train_cfg.seed, # Pass seed to trainer if it needs it internally
        batch_size=train_cfg.batch_size,
    )
    # The optimizer is now part of the PPOTrainer
    trainer = PPOTrainer(actor_critic, ppo_config)
    # Initialize optimizer state using trainer's optimizer and initial model params
    initial_opt_state = trainer.optimizer.init(initial_variables['params'])
    logger.info("Optimizer initialized.")

    # --- Orbax Checkpoint Manager Setup ---
    mngr_options = ocp.CheckpointManagerOptions(
        max_to_keep=train_cfg.orbax_max_to_keep,
        create=True # Create checkpoint dir if it doesn't exist
    )
    checkpointer = ocp.StandardCheckpointer() # Or AsyncCheckpointer
    mngr = ocp.CheckpointManager(
        directory=os.path.abspath(train_cfg.checkpoint_dir), # Use absolute path
        checkpointers=checkpointer,
        options=mngr_options
    )
    logger.info(f"Orbax CheckpointManager initialized at {train_cfg.checkpoint_dir}")


    # --- Restore Checkpoint or Initialize State ---
    # Define the structure of the state to be saved/restored
    initial_train_state = TrainState.create(
        apply_fn=actor_critic.apply, # Required by flax TrainState
        params=initial_variables['params'],
        tx=trainer.optimizer, # Store the optimizer transformation
        rng=state_rng_key, # Use a dedicated RNG for the state
        global_step=0,
        update_count=0
        # Add other non-static model variables if needed (e.g., batch stats)
        # variables=initial_variables # If you need the full variable dict (incl. non-params)
    )

    # Attempt to restore the latest checkpoint
    latest_step = mngr.latest_step()
    if latest_step is not None:
        logger.info(f"Attempting to restore checkpoint from step {latest_step}...")
        # Restore using the initial_train_state as the target structure
        restored_state = mngr.restore(
            latest_step,
            args=ocp.args.StandardRestore(abstract_ckpt=initial_train_state) # Provide structure
        )
        if restored_state:
            train_state = restored_state
            # Extract RNG key needed for trainer (if trainer manages its own key)
            # trainer_key = train_state.rng # Overwrite if rng is part of state
            logger.info(f"Successfully restored state from step {latest_step} (Global Step: {train_state.global_step}, Update: {train_state.update_count}).")
        else:
            logger.warning(f"Failed to restore checkpoint from step {latest_step}. Using initial state.")
            train_state = initial_train_state
    else:
        logger.info("No existing checkpoint found. Starting with initial state.")
        train_state = initial_train_state

    # --- Training Loop ---
    logger.info(f"Starting training loop from Global Step: {train_state.global_step}, Update: {train_state.update_count}")
    logger.info(f"Target total timesteps: {train_cfg.total_timesteps}")
    start_time = time.time()

    # Use TQDM for progress bar based on total_timesteps
    # Initial value should reflect the restored global step
    pbar = tqdm.tqdm(
        initial=train_state.global_step,
        total=train_cfg.total_timesteps,
        desc="Training Progress"
    )
    last_logged_global_step = train_state.global_step # Track for SPS calculation

    while train_state.global_step < train_cfg.total_timesteps:
        current_update_count = train_state.update_count + 1

        # Construct the state dict expected by the trainer (might need adjustment)
        # Assuming trainer needs 'params' and 'opt_state' directly
        # And maybe the RNG key if it doesn't manage its own state
        # trainer_input_state = {
        #     "variables": {'params': train_state.params}, # Pass only params if that's what train_step expects
        #     "opt_state": train_state.opt_state,
        #     "rng": trainer_key # Pass the separate trainer RNG key
        # }

        # Perform one training step (rollout + update)
        # PPOTrainer.train_step expects (params, opt_state) and returns (new_params, new_opt_state, metrics, updated_rng)
        new_params, new_opt_state, metrics, updated_trainer_rng = trainer.train_step(
            train_state.params,       # Pass params directly
            train_state.opt_state       # Pass optimizer state
            # No RNG key needed here, trainer manages its own internal RNG
        )

        # Update the trainer's RNG key for the next step
        trainer_key = updated_trainer_rng

        # Calculate steps taken in this update
        steps_this_update = metrics["episode_length"].item() # .item() to get scalar

        # Update the main TrainState
        train_state = train_state.replace(
            params=new_params, # Update params directly from train_step output
            opt_state=new_opt_state,
            # rng=new_rng, # Don't update state's RNG here, trainer manages its own
            global_step=train_state.global_step + steps_this_update,
            update_count=current_update_count
        )

        # Update progress bar
        pbar.update(steps_this_update)
        pbar.set_postfix({"Return": f"{metrics['episode_return'].item():.2f}"})

        # --- Logging ---
        if current_update_count % train_cfg.log_interval == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Calculate SPS based on steps since last log
            steps_since_last_log = train_state.global_step - last_logged_global_step
            sps = steps_since_last_log / elapsed_time if elapsed_time > 0 else 0
            start_time = end_time # Reset timer for next interval
            last_logged_global_step = train_state.global_step # Update last logged step

            log_data = {
                "update": current_update_count,
                "global_step": train_state.global_step,
                "sps": sps,
                **{k: v.item() for k, v in metrics.items()} # Convert JAX arrays to scalars
            }
            logger.info(
                f"Update: {current_update_count}, Global Step: {train_state.global_step}, "
                f"SPS: {sps:.2f}, Return: {metrics['episode_return']:.2f}"
            )
            wandb.log(log_data, step=train_state.global_step)

        # --- Orbax Model Saving ---
        if current_update_count % train_cfg.save_interval_updates == 0:
            logger.info(f"Saving checkpoint for update {current_update_count} (Global Step: {train_state.global_step})")
            # Save the entire TrainState object
            save_args = ocp.args.StandardSave(train_state)
            mngr.save(
                step=train_state.global_step, # Use global_step for checkpoint naming
                args=save_args,
                force=True # Overwrite if a checkpoint for this step exists (optional)
            )
            # Optionally wait for save to complete if using AsyncCheckpointer
            # mngr.wait_until_finished()
            logger.info(f"Checkpoint saved successfully at step {train_state.global_step}.")


    pbar.close() # Close the progress bar
    logger.info("Training finished.")

    # --- Final Orbax Save ---
    logger.info(f"Saving final model state at Global Step: {train_state.global_step}")
    save_args = ocp.args.StandardSave(train_state)
    mngr.save(
        step=train_state.global_step,
        args=save_args,
        force=True # Overwrite potentially existing checkpoint at this step
    )
    mngr.wait_until_finished() # Wait for the final save to complete
    logger.info(f"Final model state saved successfully to {mngr.directory}/.")


    wandb.finish()

if __name__ == "__main__":
    main() 