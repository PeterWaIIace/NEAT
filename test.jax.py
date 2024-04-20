import numpy as np
import jax.numpy as jnp
import time

iterations = 20000
array1 = jnp.array([[1,2], [2,5], [3,4], [4,1], [5,2]])

print(array1[array1[:,1] == 2])
