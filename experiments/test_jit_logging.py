import jax
from jax import jit
from my_jit_module import add
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

    jitted_add = jit(add)
    print("first run")
    jitted_add(1, 2)

    print("second run")
    jitted_add(1, 2)

    print("third run")
    jitted_add(1, 2)


