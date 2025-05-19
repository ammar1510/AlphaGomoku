import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

def main():
    # 1. Setup devices and mesh
    num_devices = 4
    if len(jax.devices()) < num_devices:
        raise ValueError(f"This example requires at least {num_devices} devices.")
    
    # Use the first `num_devices` devices
    devices = jax.devices()[:num_devices]
    # Create a 1D mesh, mapping device axis 'x' to the available devices
    mesh = Mesh(devices, axis_names=('x',))
    print(f"Using {len(devices)} devices: {devices}")
    print(f"Created mesh with shape: {mesh.shape}")

    # 2. Define data sharding
    # PartitionSpec('x') means shard the first axis of the array across the 'x' dimension of the mesh
    data_sharding = NamedSharding(mesh, P('x'))
    # Parameters will be replicated across all devices
    replicated_sharding = NamedSharding(mesh, P()) 

    # 3. Create and shard sample data
    batch_size = 32
    feature_dim = 8
    
    # Global data (lives on host initially)
    global_features = np.random.randn(batch_size, feature_dim).astype(np.float32)
    global_targets = np.random.randn(batch_size, 1).astype(np.float32)

    # Shard the data across devices using jax.device_put
    # The first dimension (batch) is sharded across the 'x' mesh axis
    sharded_features = jax.device_put(global_features, data_sharding)
    sharded_targets = jax.device_put(global_targets, data_sharding)

    print(f"Features sharding: {sharded_features.sharding}")
    print(f"Targets sharding: {sharded_targets.sharding}")
    # You can visualize how the data is laid out (optional)
    # jax.debug.visualize_array_sharding(sharded_features)

    # 4. Define a simple linear model
    def predict(params, x):
        w, b = params
        return jnp.dot(x, w) + b

    # 5. Define a loss function (Mean Squared Error)
    def loss_fn(params, x, y):
        y_pred = predict(params, x)
        return jnp.mean((y_pred - y) ** 2)

    # 6. Define the training step
    def train_step(params, x, y, learning_rate=0.01):
        # Calculate loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        
        # Apply gradients (simple SGD) - note JAX handles the parallelism automatically
        # Because inputs (x, y) are sharded, and params are replicated,
        # jax.jit figures out how to perform the computation efficiently.
        # Gradients calculated on each device shard will be automatically summed.
        updated_params = jax.tree_map(
            lambda p, g: p - learning_rate * g, params, grads
        )
        return updated_params, loss

    # 7. JIT compile the training step
    # jax.jit optimizes the function for the sharded inputs and replicated params
    jitted_train_step = jax.jit(
        train_step, 
        # Static argnums can sometimes improve compilation time/performance
        # static_argnums=(3,) 
    )

    # 8. Initialize parameters (replicated across devices)
    key = jax.random.PRNGKey(0)
    key_w, key_b = jax.random.split(key)
    
    # Initial parameters on the host
    initial_w = jax.random.normal(key_w, (feature_dim, 1), dtype=np.float32)
    initial_b = jax.random.normal(key_b, (1,), dtype=np.float32)
    host_params = (initial_w, initial_b)

    # Replicate parameters across all devices in the mesh
    params = jax.device_put(host_params, replicated_sharding)
    print(f"Initial params sharding (w): {params[0].sharding}")


    # 9. Run training loop
    num_epochs = 10
    print("Starting training...")
    for epoch in range(num_epochs):
        params, current_loss = jitted_train_step(params, sharded_features, sharded_targets)
        # Ensure computation completes before printing loss for accurate timing/value
        current_loss.block_until_ready() 
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {current_loss:.4f}")

    print("Training finished.")
    print(f"Final params sharding (w): {params[0].sharding}")

if __name__ == "__main__":
    main()
