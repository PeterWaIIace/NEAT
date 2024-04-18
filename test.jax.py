import time
import jax.numpy as jnp
import jax.random as jrnd

superior = jnp.array([[1,4], [3,6], [6,2], [3,9], [6,2], [3,9]])
inferior = jnp.array([[2,4], [0,0], [6,7], [2,6], [0,0], [2,6]])

wiz = jnp.array([0,1,2,3,6,3,2,3]) + 9
print(wiz)

rnd_key = jrnd.PRNGKey(int(time.time()))
rnd_value = jrnd.uniform(rnd_key)
rnd_value_int = int(rnd_value  * len(superior))
indecies = jrnd.permutation(rnd_key, superior.shape[0])[:rnd_value_int]
print(rnd_key)
rnd_key = jrnd.split(rnd_key,1)[0]
print(rnd_key)
    
print(indecies)
rnd_value = jrnd.uniform(rnd_key)
rnd_value_int = int(rnd_value  * len(superior))
indecies = jrnd.permutation(rnd_key, superior.shape[0])[:rnd_value_int]
print(indecies)
print(superior[indecies])
print(inferior[indecies])
# Define the arrays

# Subtract array2 from array1 element-wise
# subtracted_array = jnp.subtract(superior, inferior)
# subtracted_array = inferior[subtracted_array[:,0] == 0]

# # Display the subtracted array and the combined array
# print("Subtracted array:", subtracted_array ,subtracted_array.size)
