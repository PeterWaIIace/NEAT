from dataclasses import dataclass, field
from src.utils import NodeTypes
from typing import List

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


class Genome:

    def __init__(self):
        self.innovation: int = 0

        self.ngenome = [
            NodeGene(NodeTypes.INPUT, 0),
            NodeGene(NodeTypes.INPUT, 1),
            NodeGene(NodeTypes.INPUT, 2),
            NodeGene(NodeTypes.OUTPUT, 3)
        ]

        self.cgenome = [
            ConnectionGene(0, 0, 0, 3, 1.0, 1),
            ConnectionGene(1, 0, 1, 3, 1.0, 1),
            ConnectionGene(2, 0, 2, 3, 1.0, 1)
        ]

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


class Evomixer:

    def __init__(self, population_size=50):

        self.population_size = population_size
        self.genomes = [Genome()] * 50

        pass

    def get_population(self):
        return self.genomes

    def compatibility_distance(self, genome_1, genome_2, c1=1.0, c2=1.0, c3=1.0, N=1.0):

        D = 0  # disjoint genes
        E = 0  # excess genes

        # check which genome is shorter:
        less_innovative_genome = genome_1 if genome_1.innovation < genome_2.innovation else genome_2
        different_genes_1_2 = set([cgene.innovation for cgene in genome_1.cgenome]) - \
            set([cgene.innovation for cgene in genome_2.cgenome])
        different_genes_2_1 = set([cgene.innovation for cgene in genome_2.cgenome]) - \
            set([cgene.innovation for cgene in genome_1.cgenome])

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
                if value_1.innovation == value_2.innovation:
                    number_of_matched_weights += 1
                    sum_weights += abs(value_1.weight - value_2.weight)

        W = sum_weights/number_of_matched_weights

        d = (D*c1)/N + (E*c2)/N + W*c3
        return d

    def mutate(self, mutate_rate=0.2):

        for genome in self.genomes:

            # mutate - add connection
            if mutate_rate > random.uniform(random.PRNGKey(0), shape=(1,)):
                genome.add_connection()

            # mutate - add node on existing connection
