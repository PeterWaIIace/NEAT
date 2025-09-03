"""Fast neat"""

import copy
import time
import random
import networkx as nx
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from tqdm import tqdm
from gymnasium.spaces import Discrete, Box
from utils.arrayPainter import display_array
from utils.random import StatefulRandomGenerator
from src.activations import activation_func, act2name, NUMBER_OF_ACTIVATION_FUNCTIONS
from src.genome import Genome, NodeTypes

# First networkx library is imported
# along with matplotlib

Rnd = StatefulRandomGenerator()

# assuming that population is class having con_gens of size N_POPULATION x CREATURE_GENES x All information
def δ(genome_1, genome_2, c1=1.0, c2=1.0, c3=1.0, N=1.0):
    """calculate compatibility between genomes"""
    D = 0  # disjoint genes
    E = 0  # excess genes

    # check smaller innovation number:
    innovation_thresh = genome_1.max_innov if genome_1.max_innov < genome_2.max_innov else genome_2.max_innov

    # Step 1: Determine the sizes of the first dimension
    size1 = genome_1.connections.shape[0]
    size2 = genome_2.connections.shape[0]

    # Step 2: Find the maximum size
    max_size = max(size1, size2)

    # Step 3: Pad the smaller array
    if size1 < max_size:
        padding = ((0, max_size - size1), (0, 0))  # Pad the first dimension
        genome_1.connections = jnp.pad(genome_1.connections, padding)
    elif size2 < max_size:
        padding = ((0, max_size - size2), (0, 0))  # Pad the first dimension
        genome_2.connections = jnp.pad(genome_2.connections, padding)

    D_tmp = jnp.subtract(
        genome_1.connections[:innovation_thresh, 0], genome_2.connections[:innovation_thresh, 0]
    )  # getting all disjoint connections - if array do not have connection then it will have 0 as innovtaion number
    D = len(D_tmp[D_tmp != 0])

    E_tmp = jnp.subtract(
        genome_1.connections[innovation_thresh:, 0], genome_2.connections[innovation_thresh:, 0]
    )  # getting all disjoint connections - if array do not have connection then it will have 0 as innovtaion number
    E = len(E_tmp[E_tmp != 0])

    W_1 = jnp.sum(genome_1.connections[:innovation_thresh, :][D_tmp == 0][genome_1.C_W])
    W_2 = jnp.sum(genome_2.connections[:innovation_thresh, :][D_tmp == 0][genome_2.C_W])
    W_avg = jnp.subtract(W_1, W_2) / W_1.size

    d = abs((c1 * E) / N + (c2 * D) / N + c3 * W_avg)
    return d


def sh(δ, δ_t=0.2):
    """sharing fitness threshold function"""
    return δ < δ_t


# population can be matrix of shapex N_SIZE_POP x
def speciate(population, δ_th=5, **kwargs) -> list:
    """function for speciation"""
    species = [[population[0]]]

    for _, individual_2 in enumerate(population):
        if individual_2 is not population[0]:
            if sh(δ(population[0], individual_2, **kwargs), δ_th):
                species[len(species) - 1].append(individual_2)

    for i, individual_1 in enumerate(population):
        # if not in current species, create new specie
        if sum([individual_1 in specie for specie in species]) == 0:
            species.append([individual_1])
            for _, individual_2 in enumerate(population):
                if sum([individual_2 in specie for specie in species]) == 0:
                    if sh(δ(individual_1, individual_2, **kwargs), δ_th):
                        species[len(species) - 1].append(individual_2)

    # print(f"[DEBUG] Number of species: {len(species)}, if same as population number then bad.")
    # print(f"[DEBUG] Population number {len(population)}")
    return species


def mate(superior: Genome, inferior: Genome):
    """mate superior Genome with inferior Genome"""
    # check smaller innovation number:
    superior.check_against(inferior)
    inferior.check_against(superior)

    innovation_thresh = superior.max_innov if superior.max_innov < inferior.max_innov else inferior.max_innov

    offspring = copy.deepcopy(inferior)

    indecies = Rnd.randint_permutations(innovation_thresh)
    offspring.connections = offspring.connections.at[indecies].set(superior.connections[indecies])

    indecies = Rnd.randint_permutations(len(inferior.connections[innovation_thresh:])) + innovation_thresh
    offspring.connections = offspring.connections.at[indecies].set(superior.connections[indecies])
    # Lazy but working, copy all nodes not existing in inferior but exisitng in superior
    offspring.nodes = offspring.nodes.at[inferior.nodes[:, inferior.index] == 0].set(
        superior.nodes[inferior.nodes[:, inferior.index] == 0]
    )

    return offspring


# this can be done better on arrays
def cross_over(population: list, population_size: int = 0, keep_top: int = 2, δ_th: float = 5, **kwargs):
    """cross over your population"""

    population_diff = 0
    if population_size == 0:
        population_size = len(population)
    else:
        population_diff = population_size - len(population)

    keep_top = int(keep_top)
    if keep_top < 2:
        keep_top = 2

    new_population = []
    # print("[DEBUG] Speciating")
    species = speciate(population, δ_th, **kwargs)
    species_list = []

    for s_n, specie in enumerate(species):
        sorted_specie = sorted(specie, key=lambda x: x.fitness, reverse=True)[:]
        top_species = sorted_specie[:keep_top]

        for keept in top_species:
            new_population.append(keept)
            species_list.append(s_n)

        for __n in range(len(sorted_specie) - keep_top):
            n = __n % keep_top
            m = n
            while m == n:
                m = random.randint(0, len(top_species) - 1)
            # print(f"mating: {top_species[n].fitness} with {top_species[m].fitness}")
            offspring = mate(top_species[n], top_species[m])
            new_population.append(offspring)
            species_list.append(s_n)
            n = random.randint(0, len(top_species) - 1)

    # if size is bigger than current population
    # fill it up equally
    # it may happen due to pruning
    for p_n in range(population_diff):
        specie = species[p_n % len(species)]
        sorted_specie = sorted(specie, key=lambda x: x.fitness, reverse=True)[:]
        top_species = sorted_specie[:keep_top]

        n = random.randint(0, len(top_species) - 1)
        m = random.randint(0, len(top_species) - 1)
        offspring = mate(top_species[n], top_species[m])
        new_population.append(offspring)
        species_list.append(s_n)

    population = []
    # print(f"[DEBUG] Population number {len(new_population)}, {species_list}")
    return new_population, species_list
