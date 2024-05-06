import jax.random as jrnd
import time
import jax

class StatefulRandomGenerator:

    def __init__(self):
        self.key = jrnd.PRNGKey(int(time.time()))

    def randint_permutations(self,val_range):
        ''' generate subarray of permutation array from defined values range '''
        rnd_value = jrnd.uniform(self.key)
        rnd_value_int = int(rnd_value * val_range)
        indecies = jrnd.permutation(self.key, val_range)[:rnd_value_int]
        self.key = jrnd.split(self.key,1)[0]
        return indecies

    def randint(self,max=100,min=0):
        rnd_value = jrnd.randint(self.key, shape=(1,), minval=min, maxval=max)
        self.key = jrnd.split(self.key,1)[0]
        return rnd_value[0]

    def uniform(self,max=1.0,min=0,shape=(1,)):
        random_float = jrnd.uniform(self.key, shape=shape, minval=min, maxval=max)
        self.key = jrnd.split(self.key,1)[0]
        if shape == (1,):
            return random_float[0]
        else:
            return random_float

    def binary(self,p = 0.5, shape=(1,)):
        binary_array = jax.random.bernoulli(self.key, p=p, shape=shape)
        self.key = jrnd.split(self.key,1)[0]
        if shape == (1,):
            return binary_array[0]
        else:
            return binary_array


