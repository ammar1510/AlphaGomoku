import jax
import jax.numpy as jnp
import optax
import wandb
import orbax.checkpoint as ocp
import flax.linen as nn
from flax.training import train_state
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Tuple, Optional
import time
import os
from functools import partial
import hydra.utils
import logging  

from alphagomoku.environments.gomoku import GomokuJaxEnv, GomokuState
from alphagomoku.models.gomoku.actor_critic import ActorCritic
from alphagomoku.policy.ppo import PPOConfig, PPOTrainer
from alphagomoku.training.rollout import run_episode, LoopState


# --- Configure Logging ---
# Get a logger for this module
logger = logging.getLogger(__name__)
# Basic configuration (can be enhanced)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set higher level for verbose libraries like absl (used by Orbax)
logging.getLogger('absl').setLevel(logging.WARNING)


# --- Training State ---
class TrainingState(train_state.TrainState):
    # Inherits apply_fn, params, tx, opt_state
    rng: jax.random.PRNGKey
    update_step: int = 0


# --- Main Training Function ---
@hydra.main(config_path="../../../config", config_name="train", version_base=None)
def train(cfg: DictConfig):
    """Runs the PPO training loop, configured by Hydra."""
    logger.info("Starting training process...")
    logger.info("Effective Hydra Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # === Initialization ===
    start_time = time.time()
    output_dir = os.getcwd()
    logger.info(f"Hydra output directory: {output_dir}")

    # Determine workspace root using Hydra's utility
    workspace_root = hydra.utils.get_original_cwd()
    logger.info(f"Detected workspace root: {workspace_root}")
    artifacts_dir = os.path.join(workspace_root, "artifacts")

    # WandB Setup
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config=wandb_config,
        mode=cfg.wandb.mode,
        dir=artifacts_dir, # Use artifacts_dir directly, path = artifacts_dir/wandb
        sync_tensorboard=True,
        reinit=True,
    )
    print(f"WandB Run URL: {run.get_url()}")

    # RNG Setup
    rng = jax.random.PRNGKey(cfg.seed)
    rng, env_rng, model_rng, train_rng = jax.random.split(rng, 4)

    # Environment Setup
    env = GomokuJaxEnv(
        B=cfg.num_envs,
        board_size=cfg.gomoku.board_size,
        win_length=cfg.gomoku.win_length,
    )

    # Model Setup
    model = ActorCritic(board_size=cfg.gomoku.board_size)
    dummy_obs = jnp.zeros(
        (1, cfg.gomoku.board_size, cfg.gomoku.board_size)
    )  # Single dummy obs for init
    dummy_player = jnp.ones((1,), dtype=jnp.int32)  # Use 1 as dummy player, shape (1,)
    model_params = model.init(model_rng, dummy_obs, dummy_player)["params"]

    # Optimizer Setup
    total_updates = cfg.num_epochs  # LR decays over the number of epochs
    lr_schedule = optax.linear_schedule(
        init_value=cfg.ppo.learning_rate,
        end_value=cfg.ppo.learning_rate
        * cfg.ppo.lr_decay_factor,  # Use configurable decay factor
        transition_steps=total_updates,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adam(
            learning_rate=lr_schedule, eps=cfg.ppo.adam_eps
        ),  # Use configurable epsilon
    )

    # Training State Setup
    tx = optimizer
    train_state_instance = TrainingState.create(
        apply_fn=model.apply, params=model_params, tx=tx, rng=train_rng, update_step=0
    )

    # PPO Trainer Setup
    ppo_config = PPOConfig(
        learning_rate=cfg.ppo.learning_rate,
        clip_eps=cfg.ppo.clip_eps,
        vf_coef=cfg.ppo.vf_coef,
        entropy_coef=cfg.ppo.entropy_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        update_epochs=cfg.ppo.update_epochs,
        num_minibatches=cfg.ppo.num_minibatches,
        seed=cfg.seed,
    )
    ppo_trainer = PPOTrainer(config=ppo_config)

    # Checkpointing Setup
    checkpoint_path = os.path.join(artifacts_dir, cfg.checkpoint_dir)
    logger.info(f"Checkpoint path: {checkpoint_path}")
    os.makedirs(checkpoint_path, exist_ok=True)
    orbax_options = ocp.CheckpointManagerOptions(
        save_interval_steps=cfg.save_frequency, max_to_keep=cfg.max_checkpoints
    )
    checkpointer = ocp.CheckpointManager(
        directory=checkpoint_path, options=orbax_options
    )

    # --- Restore Checkpoint (Optional) ---
    latest_step = checkpointer.latest_step()
    start_epoch = 0
    total_env_steps = 0  # Track total environment steps
    if latest_step is not None:
        logger.info(f"Restoring checkpoint from step {latest_step}...")
        # We need to restore the total_env_steps as well if we want accurate step counts
        # For simplicity, we'll restore the training state but restart step count
        # A more robust solution would store total_env_steps in the checkpoint

        # Restore the dictionary directly using PyTreeRestore
        restored_data = checkpointer.restore(
            latest_step, args=ocp.args.PyTreeRestore()
        )

        # Recreate the training state from the restored dictionary
        # Note: tx (optimizer structure) is recreated, opt_state is restored
        train_state_instance = TrainingState.create(
            apply_fn=model.apply,
            params=restored_data['params'],
            tx=tx,
            opt_state=restored_data['opt_state'],
            rng=restored_data['rng'],
            # update_step will be set via replace below
        )
        # Set the update step (epoch number)
        start_epoch = restored_data['update_step']
        train_state_instance = train_state_instance.replace(update_step=start_epoch)

        logger.info(f"Restored state epoch: {start_epoch}")
        # Restore total_env_steps from the checkpoint
        total_env_steps = restored_data.get('total_env_steps', 0) # Default to 0 if not found for backward compatibility
        logger.info(f"Restored total_env_steps to: {total_env_steps}")
        if total_env_steps == 0:
            logger.warning(f"Restored state, but total_env_steps counter started from 0 (either initial training or older checkpoint format).")
    else:
        logger.info("No checkpoint found, starting training from scratch.")
        train_state_instance = train_state_instance.replace(update_step=0)
        start_epoch = 0

    # --- Training Loop --- (Iterates over epochs)
    # num_updates = cfg.total_timesteps // (cfg.rollout_length * cfg.num_envs)
    # start_update = train_state_instance.update_step
    logger.info(
        f"Starting training from epoch {start_epoch} for {cfg.num_epochs - start_epoch} more epochs ({cfg.num_epochs} total epochs configured)..."
    )

    # Initialize environment state outside the loop if starting fresh or resuming
    env_state, current_obs, _ = env.reset(env_rng)

    # for update in range(start_update, num_updates):
    for epoch in range(start_epoch, cfg.num_epochs):
        iter_start_time = time.time()
        current_params = train_state_instance.params
        current_rng = train_state_instance.rng  # Use RNG from state

        # === Rollout Phase ===
        rollout_rng, current_rng = jax.random.split(current_rng)
        # Run episode returns full_trajectory, final_state (EnvState), current_rng
        full_trajectory, final_env_state, current_rng = run_episode(
            env=env,
            black_actor_critic=model,
            black_params=current_params,
            white_actor_critic=model,
            white_params=current_params,
            rng=rollout_rng,
            buffer_size=cfg.gomoku.board_size
            * cfg.gomoku.board_size,  # Set buffer size
        )

        # Determine actual steps this iteration based on the valid mask
        # Sum over time and batch dimensions, assuming valid_mask is (T, B)
        steps_this_iter = full_trajectory["valid_mask"].sum()
        total_env_steps += steps_this_iter

        # === GAE Calculation Phase ===
        # Get final observation and player from the final EnvState
        final_obs = final_env_state.boards
        final_players = final_env_state.current_players
        # Get value for the final state
        #need to improve this -> handle in rollout
        _, final_value_pred = model.apply(
            {"params": current_params}, final_obs, final_players
        )
        # Get values for all states in the buffer
        _, buffer_values_pred = jax.vmap(model.apply, in_axes=(None, 0, 0))(
            {"params": current_params},
            full_trajectory["observations"],
            full_trajectory["current_players"],
        )
        # Concatenate buffer values and final value
        all_values = jnp.concatenate(
            [buffer_values_pred, final_value_pred[None, :]], axis=0
        )
        advantages, returns = ppo_trainer.compute_gae_targets(
            rewards=full_trajectory["rewards"],
            values=all_values,
            dones=full_trajectory["dones"],
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
        )

        # === Data Preparation Phase ===
        batch_data = {
            "observations": full_trajectory["observations"],
            "actions": full_trajectory["actions"],
            "logprobs_old": full_trajectory["logprobs"],
            "advantages": advantages,
            "returns": returns,
            "current_players": full_trajectory["current_players"],  
            "valid_mask": full_trajectory["valid_mask"],
        }

        # Prepare batch for update (reshape T,B,... -> T*B,...)
        prepared_batch = PPOTrainer._prepare_batch_for_update(batch_data)

        # === Update Phase ===
        update_rng, current_rng = jax.random.split(current_rng)
        update_rng, updated_params, updated_opt_state, update_metrics = (
            PPOTrainer.update_step(
                rng=update_rng,
                model=model,
                params=current_params,
                optimizer=tx,
                opt_state=train_state_instance.opt_state,
                full_batch=prepared_batch,
                config=ppo_config,
            )
        )

        # Update the training state
        train_state_instance = train_state_instance.replace(
            params=updated_params,
            opt_state=updated_opt_state,
            rng=current_rng,
            update_step=epoch + 1,  # Update step now corresponds to epoch
        )

        # === Logging Phase ===
        iter_end_time = time.time()
        sps = (
            steps_this_iter / (iter_end_time - iter_start_time)
            if iter_end_time > iter_start_time and steps_this_iter > 0
            else 0
        )

        if update_metrics:
            log_data = {
                "train/epoch": epoch + 1,
                "train/total_env_steps": total_env_steps,
                "train/sps": sps,
                "train/duration_s": iter_end_time - iter_start_time,
                "train/steps_this_epoch": steps_this_iter,
                "ppo/total_loss": update_metrics.get("total_loss", jnp.nan),
                "ppo/policy_loss": update_metrics.get("policy_loss", jnp.nan),
                "ppo/value_loss": update_metrics.get("value_loss", jnp.nan),
                "ppo/entropy": update_metrics.get("entropy", jnp.nan),
                "ppo/approx_kl": update_metrics.get("approx_kl", jnp.nan),
                "ppo/clip_fraction": update_metrics.get("clip_fraction", jnp.nan),
                "ppo/mask_sum_fraction": update_metrics.get(
                    "mask_sum_fraction", jnp.nan
                ),
                "rollout/avg_episode_length": jnp.mean(
                    jnp.where(
                        full_trajectory["termination_step_indices"]
                        != jnp.iinfo(jnp.int32).max,
                        full_trajectory["termination_step_indices"],
                        jnp.nan,
                    )
                ),
            }
            wandb.log(log_data, step=total_env_steps)
        else:
            log_data = {
                "train/epoch": epoch + 1,
                "train/total_env_steps": total_env_steps,
                "train/sps": sps,
                "train/duration_s": iter_end_time - iter_start_time,
                "train/steps_this_epoch": steps_this_iter,
                "info/update_skipped": 1,
            }
            wandb.log(log_data, step=total_env_steps)

        # === Checkpointing Phase ===
        current_update_step = (
            train_state_instance.update_step
        )  # This is now the epoch number
        # Checkpoint based on epoch frequency
        if (
            current_update_step % cfg.save_frequency == 0
            and current_update_step > start_epoch
        ):
            logger.info(
                f"Saving checkpoint at epoch {current_update_step} (total env steps ~{total_env_steps})..."
            )
            # Create a dictionary containing the state to save
            save_data = {
                "params": train_state_instance.params,
                "opt_state": train_state_instance.opt_state,
                "rng": train_state_instance.rng,
                "update_step": train_state_instance.update_step,
                "total_env_steps": total_env_steps, # Add total_env_steps here
            }
            # Save the dictionary using PyTreeSave
            save_args = ocp.args.PyTreeSave(save_data)
            checkpointer.save(current_update_step, args=save_args, force=True)
            logger.info("Checkpoint saved.")

    # --- Final Cleanup ---
    logger.info("Waiting for checkpointer...")
    checkpointer.wait_until_finished()
    checkpointer.close()
    logger.info("Closing WandB run...")
    wandb.finish()
    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time:.2f} seconds.")


train()
