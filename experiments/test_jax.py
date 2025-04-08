import jax
import jax.numpy as jnp

size = 5
arr = jnp.zeros((size, size, size))

rows, cols = jnp.arange(size), jnp.arange(5)

actions = jnp.arange(size)

cur_player = jnp.array([-1, 1, 1, -1, -1])


@jax.jit
def f():
    return arr.at[rows, cols, actions].set(cur_player)


out = f()
print(out)
