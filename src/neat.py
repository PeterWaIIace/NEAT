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


class Node:

    def __init__(self, index, type, bias, act):
        self.index = index
        self.type = type
        self.bias = bias
        self.act = act
        self.layer = 1

        self.weights = []
        self.inputs = []
        if self.type == NodeTypes.INPUT.value:
            self.layer = 0
            self.weights = [1.0]

        if self.type == NodeTypes.OUTPUT.value:
            self.layer = LAST_LAYER

    def add_input(self, input_node, input_weight):
        if input_node not in self.inputs:
            self.weights.append(input_weight)
            self.inputs.append(input_node)
            for node in self.inputs:
                if node.layer >= self.layer:
                    self.layer = node.layer + 1

    def rm_input(self, input_node):
        index = self.inputs.index(input_node)
        self.weights.pop(index)
        self.inputs.pop(index)


def graph(nodes, n, rows, cols):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    node_color_map = {
        NodeTypes.INPUT.value: "skyblue",
        NodeTypes.NODE.value: "lightgreen",
        NodeTypes.OUTPUT.value: "salmon",
    }
    for node in nodes:
        index = int(node.index)
        G.add_node(index, color=node_color_map[int(node.type)])

    # Add edges to the graph based on inputs
    for node in nodes:
        for input in node.inputs:
            index = int(node.index)
            input = int(input.index)
            G.add_edge(input, index)

    # Draw the graph
    # Create subplot
    ax = plt.subplot(rows, cols, n + 1)  # ← this makes subplot n
    pos = nx.spring_layout(G)  # Layout for better visualization
    node_colors = [G.nodes[i]["color"] for i in G.nodes]

    nx.draw(G, pos, ax=ax, with_labels=True, node_size=10, node_color=node_colors, font_size=2, font_weight="bold")
    ax.set_title(f"Graph {n}")
    plt.pause(0.01)


def graphProcessor(genome):
    """compile your network into FF network"""
    # I need to make sure that all output neurons are at the same layer
    ngenomes, cgenomes = genome.nodes, genome.connections
    nodes = []
    active_nodes = ngenomes[ngenomes[:, Genome.N_EN] != 0.0]
    for _, node in enumerate(active_nodes):
        nodes.append(Node(node[Genome.N_INDEX], node[Genome.N_TYPE], node[Genome.N_BIAS], node[Genome.N_ACT]))

    for c in cgenomes[cgenomes[:, Genome.enabled] != 0.0]:
        nodes[int(c[Genome.C_OUT])].add_input(nodes[int(c[Genome.C_IN])], c[Genome.C_W])

    changes = True
    while changes:
        changes = False
        used_nodes = []
        for node in nodes:
            if len(node.inputs) > 0 or node.type != NodeTypes.NODE.value:
                used_nodes.append(node)
            else:
                changes = True
        nodes = used_nodes

        for node in used_nodes:
            nodes_to_remove = []
            for input in node.inputs:
                if input not in used_nodes:
                    nodes_to_remove.append(input)

            for rm_node in nodes_to_remove:
                node.rm_input(rm_node)
                changes = True

        # for node in used_nodes:
        #     print(f"{[int(input.index) for input in node.inputs],int(node.index)}")
        # print("-----------")
    # nodes = topologicalSort(nodes)
    return nodes


def visit(visited, node, sorted):
    """visit next nodes"""
    visited[int(node.index)] = True

    for n_node in node.inputs:
        if not visited[int(n_node.index)]:
            visit(visited, n_node, sorted)

    sorted.insert(0, node)


def topologicalSort(nodes):
    """execute topological sort on nodes"""

    visited = [False] * len(nodes)
    sorted = []

    for node in reversed(nodes):
        if not visited[int(node.index)]:
            visit(visited, node, sorted)
    return sorted


def compiler(nodes, input_size, genome):
    """compile your network into FF network"""
    # I need to make sure that all output neurons are at the same layer

    ff = FeedForward(input_size, genome)
    ff.add_neurons(nodes)

    ff.compile()
    return ff


LAST_LAYER = 0xDEADBEEF


class Layer:

    def __init__(self, index, input_size, max_width):
        self.weights = None
        self.index = index
        self.width = max_width
        self.input_size = input_size
        self.neurons = []
        self.neurons_index_offset = 0

    def update_size(self, input_size):
        self.input_size = input_size
        self.width = input_size + len(self.neurons)
        self.neurons_index_offset = input_size

    def add_neuron(self, neuron):
        neuron.in_layer = len(self.neurons)
        self.neurons.append(neuron)
        self.width += 1
        # what to do if there is more neurons than max length

    def compile(self, last=False):

        self.residual_connection = jnp.identity(self.input_size)
        if last == True:
            self.width = self.input_size
            weights = jnp.zeros((self.width, len(self.neurons)))
            self.bias = jnp.zeros(len(self.neurons))
            self.acts = jnp.zeros(len(self.neurons), dtype=jnp.int32)
        elif self.index == 0:
            weights = jnp.identity(len(self.neurons))
            self.bias = jnp.zeros(len(self.neurons))
            self.acts = jnp.zeros(len(self.neurons), dtype=jnp.int32)
        else:
            if self.input_size == 0:
                self.input_size = 1
            tmp_weights = jnp.zeros((self.input_size, len(self.neurons)))
            if self.residual_connection.shape != (0, 0):
                weights = jnp.concatenate((self.residual_connection, tmp_weights), axis=1)
            else:
                weights = tmp_weights
            self.bias = jnp.zeros((self.width))
            self.acts = jnp.zeros((self.width), dtype=jnp.int32)

        # display_array([weights,self.bias],["blue","green"])

        for n, neuron in enumerate(self.neurons):
            # update all neurons indexes based on offset in this layer
            neuron.in_layer += self.neurons_index_offset

            if len(neuron.inputs) > 0:
                column = jnp.zeros((self.input_size))
                inputs = jnp.array([in_neuron.in_layer for in_neuron in neuron.inputs], dtype=jnp.int32)
                n_weights = jnp.array(neuron.weights)

                # display_array([weights,column,inputs],["purple","yellow","orange"])
                column = column.at[inputs].set(n_weights)
                if last == True:
                    weights = weights.at[:, n].set(column)
                else:
                    weights = weights.at[:, n + self.input_size].set(column)
            if last == True:
                self.bias = self.bias.at[n].set(neuron.bias)
                self.acts = self.acts.at[n].set(int(neuron.act))
            else:
                self.bias = self.bias.at[n + self.input_size].set(neuron.bias)
                self.acts = self.acts.at[n + self.input_size].set(int(neuron.act))

        self.weights = weights
        if self.index == 0:
            self.weights = self.weights.T

        # display_array([weights,self.bias],["blue","green"])

        return self.bias.shape[0]


class FeedForward:

    def __init__(self, input_size, genome):
        self.genome = genome
        self.INPUT_SIZE = input_size
        self.max_width = input_size

        self.layers = []
        self.neurons = []
        self.layers.append(Layer(0, 0, 0))
        self.graph = nx.DiGraph()

    def dump_genomes(self):
        return {"nodes": self.genome.nodes, "connect": self.genome.connections}

    def add_neurons(self, neurons):
        self.neurons = neurons
        self.max_width = self.INPUT_SIZE
        sorted_neurons = sorted(neurons, key=lambda neuron: neuron.layer)

        output_neurons = [neuron for neuron in sorted_neurons if neuron.layer == LAST_LAYER]
        sorted_neurons = [neuron for neuron in sorted_neurons if neuron.layer != LAST_LAYER]

        for neuron in sorted_neurons:
            if len(self.layers) > neuron.layer:
                self.layers[neuron.layer].add_neuron(neuron)
            else:
                for n in range(neuron.layer - (len(self.layers) - 1)):
                    self.layers.append(Layer(len(self.layers) + n, self.INPUT_SIZE, self.max_width))

                self.layers[neuron.layer].add_neuron(neuron)

            if len(self.layers[neuron.layer].neurons) > self.max_width:
                self.max_width = len(self.layers[neuron.layer].neurons)

        last_layer = len(self.layers)
        self.layers.append(Layer(last_layer, self.INPUT_SIZE, self.max_width))
        for neuron in output_neurons:
            self.layers[last_layer].add_neuron(neuron)

    def compile(self):
        new_input = 0
        for layer in self.layers:
            layer.width = self.max_width
            if layer == self.layers[-1]:
                layer.update_size(new_input)
                layer.compile(last=True)
            else:
                layer.update_size(new_input)
                new_input = layer.compile()

    def activate(self, x):
        output_values = x
        # print("==============================================")
        for layer in self.layers:
            # display_array([output_values,layer.weights,layer.bias],["blue","red","green"])
            output_values = jnp.dot(output_values, layer.weights) + layer.bias
            output_values = activation_func(output_values, layer.acts)
        return output_values

    def dry(self):
        print("----------------------------------------")
        for layer in self.layers:
            display_array([layer.weights, layer.bias], ["red", "green"])
            print(layer.weights)
            print(layer.bias)


class NEAT:

    def __init__(
        self,
        inputs=2,
        outputs=1,
        population_size=10,
        keep_top=2,
        nmc=0.7,
        cmc=0.7,
        wmc=0.7,
        bmc=0.7,
        amc=0.7,
        C1=1.0,
        C2=1.0,
        C3=1.0,
        N=1.0,
        δ_th=5.0,
    ):
        """
        nmc - node mutation chance (0.0 - 1.0 higher number means lower chance)
        cmc - connection mutation chance (0.0 - 1.0 higher number means lower chance)
        wmc - weigth mutation chance (0.0 - 1.0 higher number means lower chance)
        bmc - bias mutation chance (0.0 - 1.0 higher number means lower chance)
        amc - activation mutation chance (0.0 - 1.0 higher number means lower chance)

        Speciation parameters:
        C1 - disjointed genese signifance modfier (default 1.0)
        C2 - extended genese signifance modfier (default 1.0)
        C3 - averge weigths difference modifier (defautl 1.0)
        N - population size signficance modifier (default 1.0)
        δ_th - threshold for making new species
        """
        self.innov = 0
        self.index = 1
        self.inputs = inputs
        self.outputs = outputs
        self.population_size = population_size
        self.population = []
        self.species = []

        self.keep_top = keep_top
        self.nmc = nmc  # node mutation chance
        self.cmc = cmc  # connection mutation chance
        self.wmc = wmc  # weight mutation chance
        self.bmc = bmc  # bias mutation chance
        self.amc = amc  # activation mutation chance

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.N = N
        self.δ_th = δ_th

        self.__population_maker()

    def __population_maker(self):
        print("[INFO] Preparing population\n")
        for n in tqdm(range(self.population_size)):
            genome = Genome()
            self.population.append(genome)

            index = 0
            for i in range(self.inputs):
                self.population[n].add_node(NodeTypes.INPUT.value, 0.0, Rnd.randint(max=NUMBER_OF_ACTIVATION_FUNCTIONS))
                index += 1

            output_offset = index
            for o in range(self.outputs):
                self.population[n].add_node(
                    NodeTypes.OUTPUT.value, 0.0, Rnd.randint(max=NUMBER_OF_ACTIVATION_FUNCTIONS)
                )
                index += 1

            creation_innov = 0
            for i in range(self.inputs):
                for o in range(self.outputs):
                    creation_innov = self.population[n].add_r_connection(creation_innov)
                    self.innov = creation_innov
            self.species.append(0)
            genome.specie = 0

    def mutate_activation(self, amc=0.7):
        for genome in self.population[self.keep_top :]:
            length = len(genome.nodes[:, genome.index])
            genome.change_activation(
                Rnd.randint(NUMBER_OF_ACTIVATION_FUNCTIONS, 0) * Rnd.binary(p=amc, shape=(length,))
            )

    def mutate_weight(self, epsylon=0.1, wmc=0.7):
        for genome in self.population[self.keep_top :]:
            length = len(genome.connections[:, genome.C_INNOV])
            genome.change_weigth(
                Rnd.uniform(max=epsylon, min=-epsylon, shape=(length,)) * Rnd.binary(p=wmc, shape=(length,))
            )

    def mutate_bias(self, epsylon=0.1, bmc=0.7):
        for genome in self.population[self.keep_top :]:
            length = len(genome.nodes[:, genome.index])
            genome.change_bias(Rnd.uniform(epsylon, -epsylon, shape=(length,)) * Rnd.binary(p=bmc, shape=(length,)))

    def mutate_nodes(self, nmc=0.7):
        for genome in self.population[self.keep_top :]:
            mutation_chance = random.randint(0, 10)
            if mutation_chance > nmc * 10:
                self.innov = genome.add_r_node(self.innov)

    def mutate_connections(self, cmc=0.7):
        for genome in self.population[self.keep_top :]:
            mutation_chance = random.randint(0, 10)
            if mutation_chance > cmc * 10:
                self.innov = genome.add_r_connection(self.innov)

    def cross_over(self, δ_th=5, c1=1.0, c2=1.0, c3=1.0, N=1.0):
        # print(f"[CROSS_OVER]{len(self.population)}")
        self.population, self.species = cross_over(
            self.population,
            keep_top=self.keep_top,
            population_size=self.population_size,
            δ_th=δ_th,
            c1=c1,
            c2=c2,
            c3=c3,
            N=N,
        )

        for n, _ in enumerate(self.population):
            self.population[n].specie = self.species[n]

    def prune(self, threshold):
        print(f"[PRUNE]")
        self.population = [genome for genome in self.population if threshold < genome.fitness]

    def evaluate(self):
        """function for evaluating genomes into ff networks"""
        networks = []
        plt.clf()
        for n, genome in enumerate(self.population):
            nodes = graphProcessor(genome)
            graph(nodes, n, int(self.population_size / 5), 5)
            networks.append(compiler(nodes, self.inputs, genome))
        plt.tight_layout()
        plt.pause(0.01)
        return networks

    def update(self, fitness):
        """Function for updating fitness"""
        for n, _ in enumerate(self.population):
            self.population[n].fitness = fitness[n]

    def get_params(self):
        params = []
        for n, genome in enumerate(self.population):
            params.append(
                {
                    "net": n,
                    "specienumber": genome.specie,
                    "fitness": genome.fitness,
                    "connections": len(genome.connections[genome.connections[:, genome.enabled] != 0]),
                    "nodes": len(genome.nodes[genome.nodes[:, genome.index] != 0]),
                    "nmc": self.nmc,
                    "cmc": self.cmc,
                    "wmc": self.wmc,
                    "bmc": self.bmc,
                    "amc": self.amc,
                    "C1": self.C1,
                    "C2": self.C2,
                    "C3": self.C3,
                    "N": self.N,
                }
            )
        return params


class PNEAT:

    def __init__(self):
        self.rewards = []
        self.env = None

        self.N = 20
        self.MATCHES = 3
        self.GENERATIONS = 100
        self.POPULATION_SIZE = 20
        self.NMC = 0.9
        self.CMC = 0.9
        self.WMC = 0.9
        self.BMC = 0.9
        self.AMC = 0.9
        self.δ_th = 5
        self.MUTATE_RATE = 1
        self.RENDER_HUMAN = True
        self.epsylon = 0.2
        self.INPUT_SIZE = 4
        self.OUTPUT_SIZE = 2

        self.neat = None

        self.num_timesteps = 0
        self.network = None

        # for stats only
        self.epochs = 0
        self.all_eps_rewards = []
        self.all_eps_times = []

    def __create(self):

        if not self.env:
            return

        # if len(self.env.observation_space.shape) == 1:
        if isinstance(self.env.action_space, Discrete):
            self.OUTPUT_SIZE = self.env.action_space.n
        else:
            print(self.env.action_space, type(self.env.action_space), isinstance(self.env.action_space, Discrete))
            self.OUTPUT_SIZE = self.env.action_space.shape[1]

        self.INPUT_SIZE = self.env.observation_space.shape[0]
        self.neat = NEAT(
            self.INPUT_SIZE,
            self.OUTPUT_SIZE,
            self.POPULATION_SIZE,
            keep_top=4,
            nmc=0.5,
            cmc=0.5,
            wmc=0.5,
            bmc=0.5,
            amc=0.5,
            N=self.N,
            δ_th=self.δ_th,
        )

    def __mutate(self):
        # EVOLVE EVERYTHING
        self.neat.cross_over(δ_th=self.δ_th, N=self.N)
        self.neat.mutate_weight(epsylon=self.epsylon, wmc=self.WMC)
        self.neat.mutate_bias(epsylon=self.epsylon, bmc=self.BMC)
        for _ in range(self.MUTATE_RATE):
            self.neat.mutate_activation(amc=self.AMC)
            self.neat.mutate_nodes(nmc=self.NMC)
            self.neat.mutate_connections(cmc=self.CMC)
        return self.neat

    def set_env(self, env):
        self.env = env

    def print_stats(self):
        print(
            f"-----------------------\n"
            f"Rollout:\n"
            f"  steps: {self.num_timesteps}\n"
            f"  epochs: {self.epochs}\n"
            f"  mean_len_eps: {np.sum(self.all_eps_times)/len(self.all_eps_times)}\n"
            f"  mean_rew_eps: {np.sum(self.all_eps_rewards)/len(self.all_eps_rewards)}\n"
            f"NEAT additional stats:\n"
            f"  Population: {len(self.neat.population)}\n"
            f"  Species: {np.max(self.neat.species) + 1}\n"
            f"-----------------------\n"
        )

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        ready_to_print = False

        print("CHECKING ENV")
        if not self.env:
            return

        print("CHECKING NEAT")
        if not self.neat:
            self.__create()

        print("CHECKING TIMESTEPS")
        if reset_num_timesteps:
            self.num_timesteps = 0

        print("HERE LEARNING")
        while self.num_timesteps < total_timesteps:
            self.rewards = []

            self.neat = self.__mutate()
            networks = self.neat.evaluate()

            print("STARTING TQDM")
            for n, network in tqdm(enumerate(networks)):
                start_time = time.time()

                total_reward = 0
                observation, _ = self.env.reset()
                done = False
                while not done:

                    actions, _ = self.predict(observation, network = network)
                    # take biggest value index and make it action performed
                    observation, reward, trunacted, terminated, info = self.env.step(np.argmax(actions))
                    self.num_timesteps += 1

                    if 0 == (self.num_timesteps % log_interval):
                        ready_to_print = True

                    total_reward += reward
                    done = trunacted or terminated

                    # TODO: add callback exectution

                self.rewards.append(total_reward)

                self.all_eps_rewards.append(total_reward)
                self.all_eps_times.append(time.time() - start_time)

            self.neat.update(self.rewards)

            if ready_to_print:
                self.print_stats()

                # reset stats trackers
                self.all_eps_rewards = []
                self.all_eps_times = []
                ready_to_print = False

            self.epochs += 1

        self.network = self.neat.evaluate()
        self.env.close()

    def predict(self, observation, state=None, mask=None, deterministic=False , network = None):

        if network is None:
            network = self.network

        if not self.env:
            return

        if not network:
            print("[WARNING] NO NETWORK FOUND, PRODUCING NULL ACTIONS")
            return np.zeros(self.OUTPUT_SIZE), None

        n_agents = 1
        if len(observation.shape) > 1:
            n_agents = observation.shape[0]

        if n_agents == 1:
            total_actions = np.array(network.activate(observation))
        else:
            total_actions = np.zeros((n_agents, self.OUTPUT_SIZE))
            for n in range(n_agents):
                actions = np.array(network.activate(observation[n]))
                total_actions[n, :] = actions
        return total_actions, None
