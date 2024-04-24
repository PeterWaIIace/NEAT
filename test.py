import jax.numpy as jnp
from jax import vmap

def __x(x):
    return x

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def ReLU(x):
    return x * (x > 0)

def LeakyReLU(x):
    α = 0.01
    return x * (x > 0) + α * x * (x <= 0)

def Softplus(x):
    return jnp.log(1 + jnp.exp(x))

def tanh(x):
    return jnp.tanh(x)

def activation_func(x,y):
    ''' branchless and vectorized activation functions'''
    return (x * int(y == 0) + sigmoid(x) * int(y == 1) + ReLU(x) * int(y == 2) + LeakyReLU(x) * int(y == 3) + Softplus(x) * int(y == 4) + tanh(x) * int(y == 5))[0]

def conditional_op(x, y):

    # Apply the corresponding operation for each pair of x and y
    return __x(x) * (y == 0) + sigmoid(x) * (y == 1) + tanh(x) * (y == 2)

# Example usage
x_values = jnp.array([1.0, 2.0, 3.0])
y_values = jnp.array([0, 1, 2])  # Corresponding y values for x values

result = vmap(conditional_op)(x_values, y_values)
print(result)
