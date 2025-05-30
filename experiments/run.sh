#! /bin/bash


export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/AlphaGomoku"

. .venv/bin/activate

python -m alphagomoku.training.train_ppo_versus