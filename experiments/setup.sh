#! /bin/sh

curl -LsSf https://astral.sh/uv/install.sh | sh

export PATH="$HOME/.local/bin:$PATH"

if ! grep -qF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
if ! grep -qF 'export PATH="$HOME/.local/bin:$PATH"' ~/.profile; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
fi

cd "$HOME/AlphaGomoku"

. .venv/bin/activate
python --version

uv pip install -e .[dev]

uv pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html