''' Fast neat '''

import jax.numpy as jnp 
import jax.random as jrnd
import random 
from enum import Enum

class NodeTypes(Enum):
    NODE = 1 
    INPUT = 2
    OUTPUT = 3 

class StatefulRandomGenerator:

    def __init__(self):
        self.key = jrnd.PRNGKey(0)

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
        return rnd_value
    
    def uniform(self,max=1.0,min=0):
        random_float = jrnd.uniform(self.key, shape=(1,), minval=min, maxval=max)
        self.key = jrnd.split(self.key,1)[0]
        return random_float
        

Rnd = StatefulRandomGenerator()

class Genome:

    def __init__(self):
        self.connections_length = 20
        self.nodes_length = 20
        self.max_innov = 0
        self.index = 0
        self.fitness = 1.0
        # helper indicies names
        self.i = 2
        self.o = 3
        self.w = 4
        self.enabled = 5

        # Connections genomes is array, rows are genomes, but cols are parameters of that genomes
        self.con_gen  = jnp.zeros((self.connections_length,6),)
        self.node_gen = jnp.zeros((self.nodes_length,2),)

    def new_fitness(self,fit : float):
        ''' Assign new fitness to this genome '''
        self.fitness = fit

    def add_node(self,index,type):
        ''' Adding node '''
        if self.nodes_length <= index:
            self.nodes_length += 20
            new_nodes_spaces = jnp.zeros((20,6),)
            self.node_gen = jnp.concatenate((self.node_gen,new_nodes_spaces), axis=1)

        self.node_gen = self.node_gen.at[index].set(jnp.array([index,type.value]))

    def add_r_connection(self,innov):
        innov+=1
        possible_input_nodes  = self.node_gen[self.node_gen[:,0] != 0 and self.node_gen[:,1] != NodeTypes.OUTPUT][:,0]
        possible_output_nodes = self.node_gen[self.node_gen[:,0] != 0 and self.node_gen[:,1] != NodeTypes.INPUT][:,0]

        in_node = possible_input_nodes[Rnd.randint(max=len(possible_input_nodes))]
        out_node = possible_output_nodes[Rnd.randint(max=len(possible_output_nodes))]
        self.add_connection(innov,in_node,out_node,1.0)

    def add_r_node(self,innov):
        self.index += 1
        exisitng_connections = self.con_gen[self.con_gen[:,0] != 0]

        existing_connection = exisitng_connections[Rnd.randint(max=len(exisitng_connections))]
        index_of_connection = existing_connection[0] - 1
        self.con_gen.at[index_of_connection].set(self.con_gen.at[index_of_connection,self.enabled].set(0.0))

        new_node = self.index

        innov+=1
        self.add_connection(innov,
                            self.con_gen.at[index_of_connection,self.i],
                            new_node,
                            self.con_gen.at[index_of_connection,self.w]
                        )
        innov+=1
        self.add_connection(innov,
                            new_node,
                            self.con_gen.at[index_of_connection,self.o],
                            self.con_gen.at[index_of_connection,self.w]
                        )
        
        return innov

    def change_weigth(self,weigth):
        exisitng_connections = self.con_gen[self.con_gen[:,0] != 0]
        existing_connection = exisitng_connections[Rnd.randint(max=len(exisitng_connections))]
        index_of_connection = existing_connection[0] - 1
        self.con_gen.at[index_of_connection].set(self.con_gen.at[index_of_connection,self.w].set(weigth))
        
    def add_connection(self,innov,in_node,out_node,weight):
        ''' Adding connection '''

        # update innovation if is bigger than current innov of genome
        if self.max_innov < innov:
            self.max_innov = innov

        if self.connections_length <= innov:
            self.nodes_length += 20
            new_connections_spaces = jnp.zeros((20,6),)
            self.con_gen = jnp.concatenate((self.con_gen,new_connections_spaces), axis=1)
        
        if self.node_gen[in_node][0] == 0:
            self.add_node(in_node,NodeTypes.NODE)

        if self.node_gen[out_node][0] == 0:
            self.add_node(out_node,NodeTypes.NODE)

        self.con_gen = self.con_gen.at[innov].set(jnp.array([innov,innov,in_node,out_node,weight,1.0]))

# assuming that population is class having con_gens of size N_POPULATION x CREATURE_GENES x All information
# For example 50 x 10 x 6 
# this could be done much faster
def δ(genome_1, genome_2, c1=1.0, c2=1.0, c3=1.0, N=1.0):
    ''' calculate compatibility between genomes '''
    D = 0  # disjoint genes
    E = 0  # excess genes

    # check smaller innovation number:
    innovation_thresh = genome_1.max_innov if genome_1.max_innov < genome_2.max_innov else genome_2.max_innov

    D_tmp = jnp.subtract(
        genome_1.con_gen[:innovation_thresh,0]
        ,genome_2.con_gen[:innovation_thresh,0]
    ) # getting all disjoint connections - if array do not have connection then it will have 0 as innovtaion number
    D = len(D_tmp[D_tmp != 0])

    E_tmp = jnp.subtract(
        genome_1.con_gen[innovation_thresh:,0]
        ,genome_2.con_gen[innovation_thresh:,0]
    ) # getting all disjoint connections - if array do not have connection then it will have 0 as innovtaion number
    E = len(E_tmp[E_tmp != 0])

    W_1 = jnp.sum(genome_1.con_gen[:innovation_thresh,:][D_tmp == 0][genome_1.w])
    W_2 = jnp.sum(genome_2.con_gen[:innovation_thresh,:][D_tmp == 0][genome_2.w])
    W_avg = jnp.subtract(W_1,W_2)/W_1.size

    d = (c1 * E) / N + (c2 * D) / N + c3 * W_avg
    return d

def sh(δ,δ_t = 0.2):
    ''' sharing fitness threshold function '''
    return δ < δ_t

# population can be matrix of shapex N_SIZE_POP x 
def speciate(population) -> list:
    """function for speciation"""
    δ_th = 10
    species = [[population[0]]]

    individual_1 = population[0]
    for _,individual_2 in enumerate(population):
        if individual_1 is not individual_2:
            if sh(δ(individual_1,individual_2),δ_th):
                species[len(species) - 1].append(individual_2)

    for _,individual_1 in enumerate(population):

        # if not in current species, create new specie
        if sum([individual_1 in specie for specie in species]) == 0:
            species.append([individual_1])
            for _,individual_2 in enumerate(population):
                if individual_1 is not individual_2:
                    if sh(δ(individual_1,individual_2),δ_th):
                        species[len(species) - 1].append(individual_2)

    return species

def mate(superior : Genome, inferior : Genome):
    ''' mate superior Genome with inferior Genome '''
    # check smaller innovation number:
    innovation_thresh = superior.max_innov if superior.max_innov < inferior.max_innov else inferior.max_innov

    offspring = jnp.copy(inferior)

    indecies = Rnd.randint_permutations(innovation_thresh)
    offspring.con_gen = offspring.con_gen.at[indecies].set(superior.con_gen[indecies])

    indecies = Rnd.randint_permutations(len(inferior.con_gen[innovation_thresh:])) + innovation_thresh
    offspring.con_gen = offspring.con_gen.at[indecies].set(superior.con_gen[indecies])
    # Lazy but working, copy all nodes not existing in inferior but exisitng in superior
    offspring.node_gen = offspring.node_gen.at[inferior.node_gen[:,0] == 0].set(superior.node_gen[inferior.node_gen[:,0] == 0])

    return offspring

# this can be done better on arrays
def cross_over(population,keep_top = 2):
    ''' cross over your population '''
    new_population = []
    species = speciate(population)

    for specie in species:
        sorted_specie = sorted(specie, key=lambda x: x.fitness, reverse=True)[:]

        org_length = len(sorted_specie)
        sorted_specie = sorted_specie[:keep_top]

        for keept in sorted_specie:
            new_population.append(keept)

        for n in range(org_length-len(new_population)):
            n = random.randint(0,len(sorted_specie))
            m = n
            while m == n:
                m = random.randint(0,len(sorted_specie))

            offspring = mate(sorted_specie[n],sorted_specie[m])
            new_population.append(offspring)

def random_mutate(population,innov = 0):
    for n,_ in enumerate(population):

        mutation_chance = random.randint(0,10)
        if mutation_chance  > 7:
            innov = population[n].add_r_node(innov)

        mutation_chance = random.randint(0,10)
        if mutation_chance  > 7:
            innov = population[n].add_r_connection(innov)

        mutation_chance = random.randint(0,10)
        if mutation_chance  > 7:
            innov = population[n].change_weigth(Rnd.uniform())
    return innov

if __name__=="__main__":
    superior = Genome()
    inferior = Genome()

    superior.add_node(0,NodeTypes.INPUT)
    superior.add_node(1,NodeTypes.INPUT)
    superior.add_node(2,NodeTypes.NODE)
    superior.add_node(3,NodeTypes.NODE)
    superior.add_node(4,NodeTypes.NODE)
    superior.add_node(5,NodeTypes.OUTPUT)

    inferior.add_node(0,NodeTypes.INPUT)
    inferior.add_node(1,NodeTypes.INPUT)
    inferior.add_node(2,NodeTypes.NODE)
    inferior.add_node(4,NodeTypes.NODE)
    inferior.add_node(5,NodeTypes.OUTPUT)

    superior.add_connection(0,0,2,1.0)
    superior.add_connection(1,1,3,1.0)
    superior.add_connection(2,0,4,1.0)
    superior.add_connection(3,1,4,1.0)
    superior.add_connection(4,2,5,1.0)
    superior.add_connection(5,3,5,1.0)
    superior.add_connection(5,4,5,1.0)

    inferior.add_connection(0,0,2,1.0)
    inferior.add_connection(3,1,4,1.0)
    inferior.add_connection(4,2,5,1.0)
    inferior.add_connection(5,4,5,1.0)

    population = [superior, inferior]
    print(speciate(population))

    # print(inferior.con_gen)
    # print(inferior.node_gen)
    # inferior = mate(superior,inferior)
    # print(inferior.con_gen)
    # print(inferior.node_gen)