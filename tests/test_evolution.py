import os
import sys
import jax 
import jax.numpy as jnp

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,f"{dir_path}/../")

from src.evo import Genome, δ, sh, EvoManager
from src.neat import compiler

def test_evolution():
    evoMan = EvoManager()

    population_size = 50
    population = [Genome()] * population_size

    # Mutate all links randomly 

    for n,individual in enumerate(population):
        population[n] = evoMan.mutate(individual)

