import jax.numpy as jnp

# Define the array
arr = jnp.array([0.21, 0.43, 0.75, 0.5, 0.49, 0.51])

# Round to the nearest integer with ties rounding to the nearest even integer
rounded_arr = jnp.round(arr + 0.5).astype(int)

print(rounded_arr)