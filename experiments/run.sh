#! /bin/bash


pid=$(ps aux | grep '[u]v run python -m alphagomoku.training.train_ppo_versus' | awk '{print $2}')

if [ -n "$pid" ]; then
    echo "Killing process $pid"
    kill -9 $pid
fi


export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="f96aad01a13c399670f72c989b19a0b7952ae6b1"


cd "$HOME/AlphaGomoku"

git switch sharding
git pull origin sharding

. .venv/bin/activate

nohup uv run python -m alphagomoku.training.train_ppo_versus &