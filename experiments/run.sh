#! /bin/bash


export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/AlphaGomoku"

git switch sharding
git pull origin sharding

. .venv/bin/activate

uv run python -m alphagomoku.training.train_ppo_versus