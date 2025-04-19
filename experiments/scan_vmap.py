import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# MRE demonstrating discrepancy between vmap(scan) vs scan(batched_fn)
# for a reverse accumulation task.

# --- Configuration ---
T = 5  # Time steps
B = 2  # Batch size
DECAY = 0.9  # Similar to gamma * lambda

# --- Generate Dummy Data ---
key = jax.random.PRNGKey(42)
key, x_key, mod_key = jax.random.split(key, 3)
xs = jax.random.uniform(x_key, (T, B))
# Modifiers (like (1-done) term, simplifying here)
modifiers = jax.random.uniform(mod_key, (T, B)) * 0.9 + 0.1  # Keep > 0

print("--- Inputs ---")
print("xs:\n", xs)
print("modifiers:\n", modifiers)

# --- Method 1: vmap over scan ---


# Scan function for a single sequence
def _scan_fn_single(carry, step_data):
    x_t, modifier_t = step_data
    # Simple accumulation: y_t = x_t + DECAY * modifier_t * y_{t+1}
    # carry here represents y_{t+1} when scanning in reverse
    new_y = x_t + DECAY * modifier_t * carry
    return new_y, new_y  # New carry is the current value


# Wrapper to scan a single sequence
def _scan_single_wrapper(xs_single, modifiers_single):
    # xs_single: (T,), modifiers_single: (T,)
    # Initial carry is 0 for the step beyond the end
    _, ys = jax.lax.scan(
        _scan_fn_single,
        0.0,  # Initial carry for single sequence
        (xs_single, modifiers_single),
        reverse=True,
    )
    return ys  # Shape (T,)


# Apply vmap to the wrapper
@jax.jit
def run_vmap_scan(xs_batch, modifiers_batch):
    # Map over the batch dimension (axis 1)
    # Input xs_batch: (T, B), modifiers_batch: (T, B)
    # Output ys_batch: (T, B)
    return jax.vmap(
        _scan_single_wrapper,
        in_axes=1,  # Map over axis 1 of inputs
        out_axes=1,  # Place results in axis 1 of output
    )(xs_batch, modifiers_batch)


result_vmap_scan = run_vmap_scan(xs, modifiers)
print("\n--- Method 1: vmap(scan) Result ---")
print(result_vmap_scan)


# --- Method 2: Direct scan on batched data ---


# Scan function designed for batched inputs
def _scan_fn_batch(carry_batch, step_data_batch):
    # carry_batch: (B,)
    # step_data_batch: (xs_batch_t, modifiers_batch_t), each shape (B,)
    xs_batch_t, modifiers_batch_t = step_data_batch
    # Element-wise calculation across the batch
    new_y_batch = xs_batch_t + DECAY * modifiers_batch_t * carry_batch
    return new_y_batch, new_y_batch  # Return carry (B,) and output slice (B,)


# Apply scan directly over the time axis (0)
@jax.jit
def run_direct_scan(xs_batch, modifiers_batch):
    # xs_batch: (T, B), modifiers_batch: (T, B)
    initial_carry_batch = jnp.zeros(xs_batch.shape[1])  # Shape (B,)
    _, ys = jax.lax.scan(
        _scan_fn_batch, initial_carry_batch, (xs_batch, modifiers_batch), reverse=True
    )
    return ys  # Shape (T, B)


result_direct_scan = run_direct_scan(xs, modifiers)
print("\n--- Method 2: Direct Scan Result ---")
print(result_direct_scan)

# --- Comparison ---
try:
    np.testing.assert_allclose(
        np.array(result_vmap_scan),  # Convert JAX arrays to NumPy for testing
        np.array(result_direct_scan),
        rtol=1e-5,
        atol=1e-8,
    )
    print("\n--- Comparison ---")
    print("Results are CLOSE (unexpected based on GAE)")
except AssertionError as e:
    print("\n--- Comparison ---")
    print("Results are DIFFERENT (as expected based on GAE):")
    print(e)
