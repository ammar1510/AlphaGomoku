import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import argparse


def calculate_entropy_coef(
    episode, initial_entropy_coef, min_entropy_coef, entropy_decay_steps
):
    """
    Calculate the entropy coefficient for a given episode using the same formula as in train.py
    """
    # Calculate progress (capped at 1.0)
    progress = min(episode / entropy_decay_steps, 1.0)

    # Apply exponential decay
    current_entropy_coef = max(
        initial_entropy_coef * (min_entropy_coef / initial_entropy_coef) ** progress,
        min_entropy_coef,
    )

    return current_entropy_coef


def load_config(config_path="cfg/train.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        raise


def plot_entropy_coefficient():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot entropy coefficient decay over training iterations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cfg/train.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--initial", type=float, help="Initial entropy coefficient (overrides config)"
    )
    parser.add_argument(
        "--min", type=float, help="Minimum entropy coefficient (overrides config)"
    )
    parser.add_argument(
        "--decay_steps", type=int, help="Entropy decay steps (overrides config)"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        help="Max episodes to plot (defaults to 2x decay_steps)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Use command line args if provided, otherwise use config values
    initial_entropy_coef = (
        args.initial if args.initial is not None else config["initial_entropy_coef"]
    )
    min_entropy_coef = args.min if args.min is not None else config["min_entropy_coef"]
    entropy_decay_steps = (
        args.decay_steps
        if args.decay_steps is not None
        else config["entropy_decay_steps"]
    )

    # Determine max episodes to plot
    max_episodes = (
        args.max_episodes if args.max_episodes is not None else 2 * entropy_decay_steps
    )

    # Generate episode numbers
    episodes = np.arange(1, max_episodes + 1)

    # Calculate entropy coefficient for each episode
    entropy_coefs = [
        calculate_entropy_coef(
            ep, initial_entropy_coef, min_entropy_coef, entropy_decay_steps
        )
        for ep in episodes
    ]

    # Create the figure and plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, entropy_coefs, "b-", linewidth=2)

    # Add a vertical line at decay_steps
    plt.axvline(
        x=entropy_decay_steps,
        color="r",
        linestyle="--",
        label=f"Decay Steps = {entropy_decay_steps}",
    )

    # Add horizontal lines for initial and min values
    plt.axhline(
        y=initial_entropy_coef,
        color="g",
        linestyle="--",
        label=f"Initial Coefficient = {initial_entropy_coef}",
    )
    plt.axhline(
        y=min_entropy_coef,
        color="m",
        linestyle="--",
        label=f"Min Coefficient = {min_entropy_coef}",
    )

    # Set titles and labels
    plt.title("Entropy Coefficient Decay Over Training Iterations", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Entropy Coefficient", fontsize=12)

    # Format y-axis to use scientific notation for small values
    if min_entropy_coef < 0.01:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    # Mark key points
    plt.scatter(
        [1, entropy_decay_steps],
        [
            initial_entropy_coef,
            calculate_entropy_coef(
                entropy_decay_steps,
                initial_entropy_coef,
                min_entropy_coef,
                entropy_decay_steps,
            ),
        ],
        color="red",
        s=50,
        zorder=5,
    )

    # Show current values from config
    plt.figtext(
        0.02,
        0.02,
        f"Config: initial={initial_entropy_coef}, min={min_entropy_coef}, decay_steps={entropy_decay_steps}",
        wrap=True,
        fontsize=9,
    )

    # Show plot
    plt.tight_layout()
    plt.savefig("entropy_coefficient_plot.png")
    print(f"Plot saved as entropy_coefficient_plot.png")
    plt.show()


if __name__ == "__main__":
    plot_entropy_coefficient()
