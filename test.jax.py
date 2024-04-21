import jax.numpy as jnp
from jax import vmap

# Define your function func(x, y)
def func(x, i, y):
    if y[i] == 0:
        return x * y
    else:
        return x

# Example arrays X and Y
X = jnp.array([1.231, 2.32, 1.54])
Y = jnp.array([0, 1, 1])
indices = jnp.arange(X.shape[0])

# Apply func to each corresponding pair of elements from X and Y
result = vmap(func, (0, 0, None))(X, indices, Y)

print(result)
