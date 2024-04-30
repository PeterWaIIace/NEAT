import jax.numpy as jnp

def create_mask(indices, length, values):
    indices_arr = jnp.array(indices)
    values_arr = jnp.array(values)
    mask = jnp.zeros(length)
    mask = mask.at[indices_arr].set(values_arr)
    return mask

indices = [3, 5, 8]
values = [0.43, 0.23, 0.54]
length = 11  # Assuming the length of the desired mask array

mask = create_mask(indices, length, values)
print(mask)
