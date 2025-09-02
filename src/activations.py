import jax.numpy as jnp

NUMBER_OF_ACTIVATION_FUNCTIONS = 6

act2name = {
    0: "x",
    1: "sigmoid",
    2: "ReLU",
    3: "LeakyReLU",
    4: "Softplus",
    5: "tanh",
}

def __x(x):
    return x

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def ReLU(x):
    return x * (x > 0)

def leakyReLU(x):
    α = 0.01
    return x * (x > 0) + α * x * (x <= 0)

def softplus(x):
    return jnp.log(1 + jnp.exp(x))

def tanh(x):
    return jnp.tanh(x)

def activation_func(x, code):
    """branchless and vectorized activation functions"""
    result = x
    result = jnp.where(code == 1, sigmoid(x), result)
    result = jnp.where(code == 2, ReLU(x), result)
    result = jnp.where(code == 3, leakyReLU(x), result)
    result = jnp.where(code == 4, softplus(x), result)
    result = jnp.where(code == 5, tanh(x), result)
    return result
