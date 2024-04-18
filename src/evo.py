from dataclasses import dataclass, field
from src.utils import NodeTypes
from typing import List

import jax.numpy as jnp
from jax import random

class NodeGene:

    def __init__(self, type, index, connected_nodes = None):
        self.type = type
        self.index = index
        if not connected_nodes:
            self.connected_nodes = set()
        else:
            self.connected_nodes = connected_nodes


class ConnectionGene:

    def __init__(self, index, innov, in_index, out_index, weight, enabled):
        self.index = index
        self.innov = innov
        self.in_neuron = in_index
        self.out_neuron = out_index
        self.weight = weight
        self.enabled = enabled


def δ( genome_1, genome_2, c1=1.0, c2=1.0, c3=1.0, N=1.0):

    D = 0  # disjoint genes
    E = 0  # excess genes

    # check which genome is shorter:
    less_innovative_genome = genome_1 if genome_1.innovation < genome_2.innovation else genome_2
    different_genes_1_2 = set([cgene.innov for cgene in genome_1.cgenome]) - \
        set([cgene.innov for cgene in genome_2.cgenome])
    different_genes_2_1 = set([cgene.innov for cgene in genome_2.cgenome]) - \
        set([cgene.innov for cgene in genome_1.cgenome])

    all_different_genes = different_genes_1_2 | different_genes_2_1

    for diffg in all_different_genes:

        if diffg < less_innovative_genome.innovation:
            D += 1
        else:
            E += 1

    sum_weights = 0
    number_of_matched_weights = 0
    for _, value_1 in enumerate(genome_1.cgenome):
        for _, value_2 in enumerate(genome_2.cgenome):
            if value_1.innov == value_2.innov:
                number_of_matched_weights += 1
                sum_weights += abs(value_1.weight - value_2.weight)

    W = sum_weights/number_of_matched_weights

    d = (D*c1)/N + (E*c2)/N + W*c3
    return d

def sh(δ,δ_t=0.2):
    return δ < δ_t

class Genome:

    def __init__(self):
        self.ngenome = [
            NodeGene(NodeTypes.INPUT, 0),
            NodeGene(NodeTypes.INPUT, 1),
            NodeGene(NodeTypes.INPUT, 2),
            NodeGene(NodeTypes.OUTPUT, 3)
        ]

        self.cgenome = [
            ConnectionGene(0, 0, 0, 3, 1.0, 1),
            ConnectionGene(1, 1, 1, 3, 1.0, 1),
            ConnectionGene(2, 2, 2, 3, 1.0, 1)
        ]

        self.innovation: int = 2

    def get_input_nodes(self):
        return [node for node in self.ngenome if node.type == NodeTypes.INPUT]

    def get_output_nodes(self):
        return [node for node in self.ngenome if node.type == NodeTypes.OUTPUT]

    def add_node(self, connection, weight):
        # remember all connected nodes so no recurrent connection can be made
        input_node = [
            gene for gene in self.ngenome if gene.index == connection.in_neuron][0]

        new_node = NodeGene(NodeTypes.HIDDEN, len(
            self.ngenome), input_node.connected_nodes)

        self.ngenome.append(new_node)
        if (
            self.add_connection(connection.in_neuron,
                                new_node.index,
                                weight)
            and
            self.add_connection(
                new_node.index,
                connection.out_neuron,
                weight)
        ):
            # disable connection:
            self.cgenome[connection.index].enabled = 0
            return True
        else:
            return False

    def add_connection(self, in_node, out_node, weight):
        connected_nodes = self.ngenome[in_node].connected_nodes

        no_forward_connection = not any(
            con.in_neuron == in_node and con.out_neuron == out_node for con in self.cgenome)
        no_recurrent_connection = not (out_node in connected_nodes)
            
        if no_forward_connection and no_recurrent_connection:
            self.innovation += 1
            # add new connected node to connected nodes
            connected_nodes.add(in_node)
            self.ngenome[out_node].connected_nodes.update(connected_nodes)
            self.cgenome.append(ConnectionGene(
                len(self.cgenome),
                self.innovation,
                in_node,
                out_node,
                weight,
                1)
            )

            return True
        else:
            return False

    def set_innovation(self, innovation):
        self.innovation = innovation

    def get_innovation(self):
        return self.innovation

    def get_possible_inputs(self):
        return [gene.index for gene in self.ngenome]

    def get_possible_nodes(self):
        return [gene.index for gene in self.ngenome if gene.type is not NodeTypes.INPUT]

class EvoManager:

    def __init__(self):
        self.innovation = 0


    def add_random_connection(self,genome):
        try_counter = 0 
        max_tries = 5
        success = False

        genome.set_innovation(self.innovation)
        while not success and try_counter < max_tries:
            min_val_in  = genome.ngenome[0].index
            max_val_in  = genome.ngenome[-1].index
            min_val_out  = 0
            max_val_out  = len(genome.get_possible_nodes())

            rand_input_index = random.randint(random.PRNGKey(0), shape=(1,), minval = min_val_in, maxval = max_val_in)[0]
            rand_output_index = random.randint(random.PRNGKey(0), shape=(1,), minval = min_val_out, maxval = max_val_out)[0]

            n_in  = genome.get_possible_inputs()[rand_input_index]
            n_out = genome.get_possible_nodes()[rand_output_index]

            rand_weight = random.uniform(random.PRNGKey(0), shape=(1,))
            success = genome.add_connection(n_in,n_out,rand_weight)

            try_counter += 1

        self.innovation = genome.get_innovation()
        return genome

    def add_random_node(self,genome):
        min_val = genome.ngenome[0].index
        max_val = genome.ngenome[-1].index
        rand_conn_index = random.randint(random.PRNGKey(0), shape=(1,), minval = min_val, maxval = max_val)[0]
        rand_weight = random.uniform(random.PRNGKey(0), shape=(1,))[0]

        genome.set_innovation(self.innovation)
        genome.add_node(genome.cgenome[rand_conn_index],rand_weight)
        self.innovation = genome.get_innovation()

        return genome

    def mutate_weight(self,genome):
        min_val = genome.ngenome[0].index
        max_val = genome.ngenome[-1].index
        rand_conn_index = random.randint(random.PRNGKey(0), shape=(1,), minval = min_val, maxval = max_val)[0]
   
        rand_weight = random.uniform(random.PRNGKey(0), shape=(1,))[0]
        genome.cgenome[rand_conn_index].weight = rand_weight

        return genome

    def mutate(self,genome, mutate_rate=0.2):

        # mutate - add connection
        if mutate_rate > random.uniform(random.PRNGKey(0), shape=(1,)):
            genome = self.add_random_connection(genome)

        if mutate_rate > random.uniform(random.PRNGKey(0), shape=(1,)):
            genome = self.add_random_node(genome)

        if mutate_rate > random.uniform(random.PRNGKey(0), shape=(1,)):
            genome = self.mutate_weight(genome)
        # mutate - add node on existing connection
        return genome
    
    def sh(self,individual_1,individual_2,δ_th):
        """fitness sharing function"""
        return (δ(individual_1,individual_2) / δ_th) < 1

    def speciate(self, population) -> list:
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

    def __mate(superior : Genome, inferior : Genome):
        key = random.PRNGKey(0)  # Initialize the random key
        # Generate random booleans (0 or 1 integers) and convert them to boolean type
        random_mating_vector_connection = random.randint(key, (len(superior.cgenome),), minval=0, maxval=2).astype(bool)
        
        for n,connections_s in enumerate(superior.cgenome):
            if random_mating_vector_connection[n]:
                for m,connections_i in enumerate(inferior.cgenome):
                    if connections_s.innov == connections_i.innov:
                        inferior.cgenome[m] = connections_s
                        if not connections_s.out_neuron in [ngenome.index for ngenome in inferior.ngenome]:
                            inferior.ngenome(NodeGene(NodeTypes.Node, connections_s.out_neuron))
                        inferior.ngenome[connections_s.out_neuron]

                    elif connections_i.innov > connections_s.innov:
                        break 



    def crossover(self,fitnesses,population):
        # do I need to adjust that fitnesses
        adjusted_fitnesses = jnp.array(fitnesses)/len(population)
        
        if len(population) == 1:
            return population
        elif len(population) == 2:

    


#### TODO:
#
# 1. Prepare test genomes for evolution
# 2. Run evolution with speciation
#
#