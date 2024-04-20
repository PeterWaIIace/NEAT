import numpy as np
import jax.numpy as jnp
import time

iterations = 20000
array1 = [1, 2, 3, 4, 5]

for n in range(iterations):
    start = time.time()
    result = [1, 2, 3, 4, 5]
    for i in range(len(result)):
        result[i] += 10
    end = time.time()

loop_time = end - start
print("For loop time:", loop_time, "array1: ", result)

array1 = np.array([1, 2, 3, 4, 5 ])
result_np = np.array([1, 2, 3, 4, 5 ])

for n in range(iterations):
    start = time.time()
    result_np = array1 + 10
    end = time.time()

vec_time = end - start
print("Vec time:     ", vec_time, "result: ", result_np)

array1 = jnp.array([1, 2, 3, 4, 5 ])

for n in range(iterations):
    start = time.time()
    result_jax = array1 + 10
    end = time.time()

jax_time = end - start
print("Jax vec time:", jax_time, "result: ", result_jax)