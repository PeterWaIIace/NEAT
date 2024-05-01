import jax.numpy as jnp

def func1(x):
    return x * 2

def func2(x):
    return x + 10

def func3(x):
    return x / 2

def func4(x):
    return x ** 2

def apply_functions(data, codes):
    # Ensure the data and codes are JAX arrays
    data = jnp.array(data)
    codes = jnp.array(codes)

    # Result array
    result = jnp.zeros_like(data, dtype=jnp.float32)

    # Apply each function based on the code
    result = jnp.where(codes == 1, func1(data), result)
    result = jnp.where(codes == 2, func2(data), result)
    result = jnp.where(codes == 3, func3(data), result)
    result = jnp.where(codes == 4, func4(data), result)

    return result

# Example data and codes
data = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
codes = jnp.array([1, 2, 2, 1, 1, 4, 3, 3, 2, 4])

# Apply functions
result = apply_functions(data, codes)
print(result)
