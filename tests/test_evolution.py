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
    population = []
    for n in range(population_size):
        g = Genome() 
        population.append(g)

    # Mutate all links randomly 
    for i in range(100):
        for n,individual in enumerate(population):
            population[n] = evoMan.mutate(individual,1.0)

    for n,individual_1 in enumerate(population):
        for n,individual_2 in enumerate(population):
            if individual_1 is not individual_2:
                assert δ(individual_1,individual_2) > 0.2        


