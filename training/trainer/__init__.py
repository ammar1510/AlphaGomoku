from training.trainer.ppo_trainer import PPOTrainer
from training.trainer.policy_gradient_trainer import PolicyGradientTrainer
# from training.trainer.self_play_trainer import SelfPlayTrainer # Commented out to avoid Gomoku dependency for Pong

__all__ = ["PPOTrainer", "PolicyGradientTrainer"] # Removed SelfPlayTrainer from __all__
