import jax
import logging

# --- Configure Logging ---
# Get a logger for this module
logger = logging.getLogger(__name__)
# Basic configuration (can be enhanced)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Set higher level for verbose libraries like absl (used by Orbax)
logging.getLogger("absl").setLevel(logging.WARNING)


print("JAX process count:", jax.process_count())
# --- Initialize JAX Distributed System ---
if jax.process_count() > 1:
    jax.distributed.initialize()
    logger.info(
            f"JAX distributed system initialized on process {jax.process_index()}/{jax.process_count()}."
        )
else:
    logger.info("JAX distributed system not initialized (single process).")