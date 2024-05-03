''' Fast neat '''
import jax
import copy
import time
import pickle
import random
import networkx as nx 
import gymnasium as gym
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from jax import jit
from enum import Enum
from arrayPainter import display_array, display_array
# First networkx library is imported  
# along with matplotlib 

class NodeTypes(Enum):
    NODE   = 1
    INPUT  = 2
    OUTPUT = 3

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
        

Rnd = StatefulRandomGenerator()

NUMBER_OF_ACTIATION_FUNCTIONS = 6

act2name = {
    0 : "x",
    1 : "sigmoid",
    2 : "ReLU",
    3 : "LeakyReLU",
    4 : "Softplus",
    5 : "tanh",
}

def __x(x):
    return x

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def ReLU(x):
    return x * (x > 0)

def leakyReLU(x):
    α = 0.01
    return x * (x > 0) + α * x * (x <= 0)

def softplus(x):
    return jnp.log(1 + jnp.exp(x))

def tanh(x):
    return jnp.tanh(x)

def activation_func(x,code):
    ''' branchless and vectorized activation functions'''
    result = x
    result = jnp.where(code == 1, sigmoid(x),  result)
    result = jnp.where(code == 2, ReLU(x),     result)
    result = jnp.where(code == 3, leakyReLU(x),result)
    result = jnp.where(code == 4, softplus(x), result)
    result = jnp.where(code == 5, tanh(x),     result)
    return result
class Genome:

    i = 2
    o = 3
    w = 4
    enabled = 5

    n_index = 0
    n_bias = 2
    n_act  = 3

    def __init__(self):
        self.index = 0
        self.specie = 0
        self.fitness = 1.0
        self.max_innov = 0
        self.nodes_length = 20
        self.connections_length = 20

        self.con_gen  = jnp.zeros((self.connections_length,6),)
        self.node_gen = jnp.zeros((self.nodes_length,4),)

    def load_genomes(self, con_gen, node_gen):
        ''' load genome by passing connection genome and node genome '''
        self.con_gen  = con_gen
        self.node_gen = node_gen
        self.max_innov = self.con_gen[self.con_gen[:,0] != 0][-1,0]
        self.connections_length = len(self.con_gen[self.con_gen[:,0] != 0])
        self.nodes_length = len(self.node_gen[self.node_gen[:,0] != 0])

    def new_fitness(self,fit : float):
        ''' Assign new fitness to this genome '''
        self.fitness = fit

    # TODO: there is problem with correct nodes
    def add_node(self,index : int, type : NodeTypes, bias : float, act : int):
        ''' Adding node '''
        if self.nodes_length <= index:
            self.nodes_length += 20
            new_nodes_spaces = jnp.zeros((20,4),)
            self.node_gen = jnp.concatenate((self.node_gen,new_nodes_spaces), axis=0)

        self.node_gen = self.node_gen.at[index].set(jnp.array([index+1,type.value,bias,act]))

    def add_r_connection(self,innov):
        active_nodes = self.node_gen[self.node_gen[:,0] != 0]
        possible_input_nodes  = active_nodes[active_nodes[:,1] != float(NodeTypes.OUTPUT.value)][:,0]
        possible_output_nodes = active_nodes[active_nodes[:,1] != float(NodeTypes.INPUT.value)][:,0]

        in_node  = possible_input_nodes[Rnd.randint(max=len(possible_input_nodes))]
        out_node = possible_output_nodes[Rnd.randint(max=len(possible_output_nodes))]
        while in_node == out_node:
            out_node = possible_output_nodes[Rnd.randint(max=len(possible_output_nodes))]
        return self.add_connection(int(innov),int(in_node)-1,int(out_node)-1,1.0)

    def add_r_node(self,innov):
        innov = int(innov)
        exisitng_connections = self.con_gen[self.con_gen[:,0] != 0]

        index_of_connection = Rnd.randint(max=len(exisitng_connections)-1)
        self.con_gen = self.con_gen.at[index_of_connection,self.enabled].set(0.0)

        new_node = len(self.node_gen[self.node_gen[:,self.n_index] != 0]) + 1

        in_node = int(self.con_gen[index_of_connection,self.i])
        out_node = int(new_node)

        innov = self.add_connection(innov,
                            in_node-1,
                            out_node-1,
                            self.con_gen[index_of_connection,self.w]
                        )

        in_node = int(new_node)
        out_node = int(self.con_gen[index_of_connection,self.o])
        innov = self.add_connection(innov,
                            in_node-1,
                            out_node-1,
                            self.con_gen[index_of_connection,self.w]
                        )
        return innov

    def add_connection(self,innov : int,in_node : int, out_node : int, weight : float):
        ''' Adding connection '''
        # update innovation if is bigger than current innov of genome
        for connections in self.con_gen[self.con_gen[:,0] != 0]:
            if out_node+1 == connections[self.i] and in_node+1 == connections[self.o]:
                return innov

        ## after this point we need to add new connection, before it we can reject it 
        innov+=1
        if self.max_innov < innov:
            self.max_innov = innov

        if self.connections_length <= innov:
            self.connections_length += 20
            new_connections_spaces = jnp.zeros((20,6),)
            self.con_gen = jnp.concatenate((self.con_gen,new_connections_spaces), axis=0)

        if self.node_gen[int(in_node),self.n_index] == 0:
            self.add_node(in_node,NodeTypes.NODE,0.0,Rnd.randint(max=NUMBER_OF_ACTIATION_FUNCTIONS,min=0))

        if self.node_gen[int(out_node),self.n_index] == 0:
            self.add_node(out_node,NodeTypes.NODE,0.0,Rnd.randint(max=NUMBER_OF_ACTIATION_FUNCTIONS,min=0))

        self.con_gen = self.con_gen.at[innov].set(jnp.array([innov,innov,in_node+1,out_node+1,weight,1.0]))
        return innov

    def change_weigth(self,weigth):
        print(f"changing weights: {weigth}")
        print(f"before: {self.con_gen[:,self.w]}")
        self.con_gen = self.con_gen.at[:,self.w].add(weigth)
        print(f"changed weights: {self.con_gen[:,self.w]}")
        
    def change_bias(self,bias):
        self.node_gen = self.node_gen.at[:,self.n_bias].add(bias)

    def change_activation(self,act):
        self.node_gen = self.node_gen.at[:,self.n_act].set(act)


# assuming that population is class having con_gens of size N_POPULATION x CREATURE_GENES x All information
# For example 50 x 10 x 6 
# this could be done much faster
def δ(genome_1, genome_2, c1=1.0, c2=1.0, c3=1.0, N=1.0):
    ''' calculate compatibility between genomes '''
    D = 0  # disjoint genes
    E = 0  # excess genes

    # check smaller innovation number:
    innovation_thresh = genome_1.max_innov if genome_1.max_innov < genome_2.max_innov else genome_2.max_innov

    # Step 1: Determine the sizes of the first dimension
    size1 = genome_1.con_gen.shape[0]
    size2 = genome_2.con_gen.shape[0]

    # Step 2: Find the maximum size
    max_size = max(size1, size2)

    # Step 3: Pad the smaller array
    if size1 < max_size:
        padding = ((0, max_size - size1), (0, 0))  # Pad the first dimension
        genome_1.con_gen = jnp.pad(genome_1.con_gen, padding)
    elif size2 < max_size:
        padding = ((0, max_size - size2), (0, 0))  # Pad the first dimension
        genome_2.con_gen = jnp.pad(genome_2.con_gen, padding)

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

    d = abs((c1 * E) / N + (c2 * D) / N + c3 * W_avg)
    return d

def sh(δ,δ_t = 0.2):
    ''' sharing fitness threshold function '''
    return δ < δ_t

# population can be matrix of shapex N_SIZE_POP x 
def speciate(population, δ_th = 5, **kwargs) -> list:
    """function for speciation"""
    species = [[population[0]]]

    for _,individual_2 in enumerate(population):
        if individual_2 is not population[0]:
            if sh(δ(population[0],individual_2,**kwargs),δ_th):
                species[len(species) - 1].append(individual_2)

    for i,individual_1 in enumerate(population):
        # if not in current species, create new specie
        if sum([individual_1 in specie for specie in species]) == 0:
            species.append([individual_1])
            for _,individual_2 in enumerate(population):
                if sum([individual_2 in specie for specie in species]) == 0:
                    if sh(δ(individual_1,individual_2,**kwargs),δ_th):
                        species[len(species) - 1].append(individual_2)

    print(f"[DEBUG] Number of species: {len(species)}, if same as population number then bad.")
    print(f"[DEBUG] Population number {len(population)}")
    return species

def mate(superior : Genome, inferior : Genome):
    ''' mate superior Genome with inferior Genome '''
    # check smaller innovation number:
    innovation_thresh = superior.max_innov if superior.max_innov < inferior.max_innov else inferior.max_innov

    offspring = copy.deepcopy(inferior)

    indecies = Rnd.randint_permutations(innovation_thresh)
    offspring.con_gen = offspring.con_gen.at[indecies].set(superior.con_gen[indecies])

    indecies = Rnd.randint_permutations(len(inferior.con_gen[innovation_thresh:])) + innovation_thresh
    offspring.con_gen = offspring.con_gen.at[indecies].set(superior.con_gen[indecies])
    # Lazy but working, copy all nodes not existing in inferior but exisitng in superior
    offspring.node_gen = offspring.node_gen.at[inferior.node_gen[:,0] == 0].set(superior.node_gen[inferior.node_gen[:,0] == 0])

    return offspring

# this can be done better on arrays
def cross_over(population : list, population_size : int = 0, keep_top : int = 2, δ_th : float = 5, **kwargs):
    ''' cross over your population '''

    population_diff = 0
    if population_size == 0:
        population_size = len(population)
    else:
        population_diff = population_size - len(population)
        print(f"[DEBUG] population_diff: {population_diff}")


    keep_top = int(keep_top)
    if keep_top < 2:
        keep_top = 2

    new_population = []
    print("[DEBUG] Speciating")
    species = speciate(population, δ_th, **kwargs)
    species_list = []

    for s_n,specie in enumerate(species):
        sorted_specie = sorted(specie, key=lambda x: x.fitness, reverse=True)[:]
        top_species = sorted_specie[:keep_top]

        for keept in top_species:
            new_population.append(keept)
            species_list.append(s_n)

        for __n in range(len(sorted_specie) - keep_top):
            n = __n % keep_top
            m = n
            while m == n:
                m = random.randint(0,len(top_species)-1)
            print(f"mating: {top_species[n].fitness} with {top_species[m].fitness}")
            offspring = mate(top_species[n],top_species[m])
            new_population.append(offspring)
            species_list.append(s_n)
            n = random.randint(0,len(top_species)-1)
    
    # if size is bigger than current population
    # fill it up equally 
    # it may happen due to pruning
    for p_n in range(population_diff):
        specie = species[p_n % len(species)]
        sorted_specie = sorted(specie, key=lambda x: x.fitness, reverse=True)[:]
        top_species = sorted_specie[:keep_top]

        n = random.randint(0,len(top_species)-1)
        m = random.randint(0,len(top_species)-1)
        offspring = mate(top_species[n],top_species[m])
        new_population.append(offspring)
        species_list.append(s_n)

    population = []
    print(f"[DEBUG] Population number {len(new_population)}, {species_list}")
    return new_population, species_list

def compiler(genome,input_size):
    ''' compile your network into FF network '''
    # I need to make sure that all output neurons are at the same layer
    ngenomes, cgenomes = genome.node_gen, genome.con_gen
    neurons = []
    active_nodes = ngenomes[ngenomes[:,0] != 0.0]
    for _,node in enumerate(active_nodes):
        neurons.append(
            Neuron(node)
        )

    for c in cgenomes[cgenomes[:,Genome.enabled] != 0.0]:
        if int(c[Genome.o])-1 < len(neurons) and int(c[Genome.i])-1 < len(neurons):
            neurons[int(c[Genome.o])-1].add_input(
                neurons[int(c[Genome.i])-1],
                c[Genome.w])
            
    ff = FeedForward(input_size,genome)
    ff.add_neurons(neurons)

    ff.compile()
    return ff

LAST_LAYER = 0xDEADBEEF
class Neuron:

    def __init__(self,node_genome):
        self.index = int(node_genome[0]) - 1
        self.in_layer = self.index
        self.layer = 0
        self.type = int(node_genome[1])
        self.bias = node_genome[Genome.n_bias]
        self.act  = node_genome[Genome.n_act]
        self.input_list = []
        self.input_neurons = []
        self.weights = []
        if self.type == NodeTypes.INPUT.value:
            self.input_list = [self.index]
            self.weights = [1.0]
   
    def add_input(self,in_neuron,weigth):
        self.input_neurons.append(in_neuron)
        self.input_list.append(in_neuron.index)
        self.weights.append(weigth)

        if self.type != NodeTypes.OUTPUT.value:
            for neuron in self.input_neurons:
                if neuron.layer >= self.layer:
                    self.layer = neuron.layer + 1
        else:
            self.layer = LAST_LAYER

    def getLayer(self):
        return self.layer

    def get(self):
        return {self.layer : jnp.array(self.weights)}

class Layer:

    def __init__(self,index,input_size,max_width):
        self.weights = None
        self.index = index
        self.width = max_width
        self.input_size = input_size
        self.neurons = []
        self.neurons_index_offset = 0
        self.vmap_activate = jax.vmap(activation_func)

    def update_size(self,input_size):
        self.input_size = input_size
        self.width = input_size + len(self.neurons)
        self.neurons_index_offset = input_size
        
    def add_neuron(self,neuron):
        neuron.in_layer = len(self.neurons)
        self.neurons.append(neuron)
        self.width += 1
        # what to do if there is more neurons than max length 

    def compile(self,last=False):
        
        self.residual_connection = jnp.identity(self.input_size)
        if last == True:
            self.width = self.input_size
            weights   = jnp.zeros((self.width,len(self.neurons)))
            self.bias = jnp.zeros(len(self.neurons))
            self.acts = jnp.zeros(len(self.neurons),dtype=jnp.int32)
        elif self.index == 0:
            weights   = jnp.identity(len(self.neurons))
            self.bias = jnp.zeros(len(self.neurons))
            self.acts = jnp.zeros(len(self.neurons),dtype=jnp.int32)
        else:
            if self.input_size == 0:
                self.input_size = 1
            tmp_weights = jnp.zeros((self.input_size,len(self.neurons)))
            if self.residual_connection.shape != (0,0):
                weights    = jnp.concatenate((self.residual_connection, tmp_weights), axis=1)
            else:
                weights = tmp_weights
            self.bias = jnp.zeros((self.width))
            self.acts = jnp.zeros((self.width),dtype=jnp.int32)
            
        for n,neuron in enumerate(self.neurons):
            # update all neurons indexes based on offset in this layer
            neuron.in_layer += self.neurons_index_offset

            if len(neuron.input_neurons) > 0:
                column = jnp.zeros((self.input_size))
                inputs = jnp.array([in_neuron.in_layer for in_neuron in neuron.input_neurons],dtype=jnp.int32)
                n_weights = jnp.array(neuron.weights)
                
                column = column.at[inputs].set(n_weights)
                if last == True:
                    weights = weights.at[:,n].set(column)
                else:
                    weights = weights.at[:,n+self.input_size].set(column)
            if last == True:
                self.bias = self.bias.at[n].set(neuron.bias)
                self.acts = self.acts.at[n].set(int(neuron.act))
            else:         
                self.bias = self.bias.at[n+self.input_size].set(neuron.bias)
                self.acts = self.acts.at[n+self.input_size].set(int(neuron.act))
        
        self.weights = weights
        if self.index == 0:
            self.weights = self.weights.T

        # display_array([self.weights, self.bias],["green","blue"])
        return self.bias.shape[0]
    
class FeedForward:

    def __init__(self,input_size,genome):
        self.genome = genome
        self.INPUT_SIZE = input_size
        self.max_width = input_size

        self.layers = []
        self.layers.append(Layer(0,0,0))
        self.graph = nx.DiGraph()

    def dump_genomes(self):
        return {"nodes":self.genome.node_gen,"connect" : self.genome.con_gen}

    def add_neurons(self,neurons):
        self.max_width = self.INPUT_SIZE
        sorted_neurons = sorted(neurons,key=lambda neuron: neuron.layer)
        

        output_neurons = [neuron for neuron in sorted_neurons if neuron.layer == LAST_LAYER]
        sorted_neurons = [neuron for neuron in sorted_neurons if neuron.layer != LAST_LAYER]

        for neuron in sorted_neurons:
            if len(neuron.input_neurons) == 0 and neuron.type != NodeTypes.INPUT.value:
                continue
            
            if len(self.layers) > neuron.layer:
                self.layers[neuron.layer].add_neuron(neuron)
            else:
                for n in range(neuron.layer - (len(self.layers)-1)):
                    self.layers.append(Layer(len(self.layers) + n,self.INPUT_SIZE,self.max_width))
                try:
                    self.layers[neuron.layer].add_neuron(neuron)
                except Exception as e:
                    print(f"Crashed here: {e} with {neuron.layer} and len {len(self.layers)}")
                    exit()

            if len(self.layers[neuron.layer].neurons) > self.max_width:
                self.max_width = len(self.layers[neuron.layer].neurons)

        last_layer = len(self.layers)
        self.layers.append(Layer(last_layer,self.INPUT_SIZE,self.max_width))
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

    def activate(self,x):
        output_values = x
        for layer in self.layers:
            output_values = jnp.dot(output_values,layer.weights) + layer.bias
            output_values = activation_func(output_values,layer.acts)
        return output_values
    
    def dry(self):
        print("----------------------------------------")
        for layer in self.layers:
            display_array([layer.weights,layer.bias],["red","green"])
            print(layer.weights)
            print(layer.bias)

    def __add_edge_to_graph(self,n1,n2,label,thickness):
        # if n1 in self.graph.nodes() and n2 in self.graph.nodes():
        self.graph.add_edge(n1,n2,label=label,thickness=thickness)

    def __add_node_to_graph(self,node,x,layer_length,max_layer = 10):

        salmon_color = mcolors.to_rgba("salmon", alpha=0.5)
        lightgreen_color = mcolors.to_rgba("lightgreen", alpha=0.5)
        skyblue_color = mcolors.to_rgba("skyblue", alpha=0.5)

        if node.type == NodeTypes.INPUT.value:
            self.graph.add_node(node.index,
                                color=salmon_color,
                                label=f"{act2name[int(node.act)]}\nbias: {node.bias:.2f}\nnode: {node.index}")
            nx.set_node_attributes(self.graph, {node.index: (x,0)}, "pos")
        if node.type == NodeTypes.OUTPUT.value:
            self.graph.add_node(node.index,
                                color=lightgreen_color,
                                label=f"{act2name[int(node.act)]}\nbias: {node.bias:.2f}\nnode: {node.index}")
            nx.set_node_attributes(self.graph, {node.index: (x,max_layer+1)}, "pos")
        if node.type == NodeTypes.NODE.value:
            self.graph.add_node(node.index,
                                color=skyblue_color,
                                label=f"{act2name[int(node.act)]}\nbias: {node.bias:.2f}\nnode: {node.index}")
            x += (node.layer)/10
            nx.set_node_attributes(self.graph, {node.index: (x, node.layer)}, "pos")

    def visualize(self,name):
        ''' Visualize graph of current network '''
        for l_n,l in enumerate(self.layers):
            for x,neuron in enumerate(l.neurons):
                # print(f"displaying node: {neuron.index}")
                if neuron.index >= 0:
                    self.__add_node_to_graph(neuron,x,layer_length=len(l.neurons),max_layer = len(self.layers))
            
            for neuron in l.neurons:
                for n_inputs,w in zip(neuron.input_list,neuron.weights):
                    # print(f"displaying connection: {n_inputs} {neuron.index}")
                    if n_inputs >= 0 and neuron.index >= 0:
                        self.__add_edge_to_graph(n_inputs, neuron.index, label=f'{w:.2f}',thickness=abs(w))

        # Get weakly connected components
        components = list(nx.weakly_connected_components(self.graph))

        # Find the largest weakly connected component
        largest_component = max(components, key=len)

        # Create a subgraph with only the largest weakly connected component
        largest_subgraph = self.graph.subgraph(largest_component)

        # Filter out isolated nodes
        isolated_nodes = [node for node in largest_subgraph.nodes() if largest_subgraph.degree(node) == 0]
        self.graph.remove_nodes_from(isolated_nodes)

        # Draw the graph with node colors and labels
        try:
            pos = {node: self.graph.nodes[node]["pos"] for node in self.graph.nodes()}
        except Exception as e:
            print(f"Caught exception: {e}")
            print({node: self.graph.nodes[node] for node in self.graph.nodes()})
            return 
        node_colors = [self.graph.nodes[node]["color"] for node in self.graph.nodes()]
        node_labels = {node: self.graph.nodes[node]["label"] for node in self.graph.nodes()}
        edge_thickness = [nx.get_edge_attributes(self.graph, 'thickness')[edge] for edge in self.graph.edges()]
        plt.clf()
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, labels=node_labels, node_size=800, font_size=5, width = edge_thickness)  # Draw nodes
        plt.savefig(f"models/{name}.png")
        self.graph.clear()
        # plt.show()

class NEAT:

    def __init__(self,inputs=2, outputs=1, population_size = 10, 
                nmc = 0.7, 
                cmc = 0.7, 
                wmc = 0.7, 
                bmc = 0.7, 
                amc = 0.7,
                C1 = 1.0,
                C2 = 1.0,
                C3 = 1.0,
                N = 1.0,
                δ_th = 5.0
                ):
        '''
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
        '''
        self.innov = 0
        self.index = 1
        self.inputs = inputs
        self.outputs = outputs
        self.population_size = population_size
        self.population = []
        self.species = []

        self.nmc = nmc# node mutation chance
        self.cmc = cmc# connection mutation chance
        self.wmc = wmc# weight mutation chance
        self.bmc = bmc# bias mutation chance
        self.amc = amc# activation mutation chance

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.N = N
        self.δ_th = δ_th

        self.__population_maker()

    def __population_maker(self):
        for n in range(self.population_size):
            genome = Genome()
            self.population.append(genome)
            
            index =  0
            for i in range(self.inputs):
                self.population[n].add_node(index,NodeTypes.INPUT,0.0,Rnd.randint(max=NUMBER_OF_ACTIATION_FUNCTIONS))
                index += 1
            
            output_offset = index
            for o in range(self.outputs):
                self.population[n].add_node(index,NodeTypes.OUTPUT,0.0,Rnd.randint(max=NUMBER_OF_ACTIATION_FUNCTIONS))
                index += 1

            creation_innov = 0
            for i in range(self.inputs):
                for o in range(self.outputs):
                    creation_innov = self.population[n].add_connection(creation_innov,i,output_offset+o,0.0)
                    self.innov = creation_innov
            self.species.append(0)
            genome.specie = 0

    def load_population(self,file_name):

        __genomes = pickle.load(open(file_name,"rb"))
        ngenom = __genomes["nodes"]
        cgenom = __genomes["connect"]

        self.population = []
        for _ in range(self.population_size):
            genome = Genome()
            genome.load_genomes(cgenom,ngenom)
            self.population.append(genome)

    def mutate_activation(self,amc = 0.7):
        for genome in self.population:
            length = len(genome.node_gen[:,0])
            genome.change_activation(
                Rnd.randint(NUMBER_OF_ACTIATION_FUNCTIONS,0) *
                Rnd.binary(p = amc,shape=(length,))
            )

    def mutate_weight(self,epsylon = 0.1,wmc = 0.7):
        for genome in self.population:
            length = len(genome.con_gen[:,0])
            genome.change_weigth(
                Rnd.uniform(max=epsylon,min=-epsylon,shape=(length,)) *
                Rnd.binary(p = wmc,shape=(length,))
            )

    def mutate_bias(self,epsylon = 0.1,bmc = 0.7):
        for genome in self.population:
            length = len(genome.node_gen[:,0])
            genome.change_bias(
                Rnd.uniform(epsylon,-epsylon,shape=(length,)) *
                Rnd.binary(p = bmc,shape=(length,))
            )

    def mutate_nodes(self,nmc = 0.7):
        for genome in self.population:
            mutation_chance = random.randint(0,10)
            if mutation_chance > nmc*10:
                self.innov = genome.add_r_node(self.innov)

    def mutate_connections(self,cmc = 0.7):
        for genome in self.population:
            mutation_chance = random.randint(0,10)
            if mutation_chance  > cmc*10:
                self.innov = genome.add_r_connection(self.innov)

    def cross_over(self,keep_top = 2, δ_th = 5, c1 = 1.0, c2 = 1.0, c3 = 1.0, N = 1.0):
        print(f"[CROSS_OVER]{len(self.population)}")
        self.population, self.species = cross_over(self.population,
                keep_top = keep_top,
                population_size = self.population_size,
                δ_th = δ_th,
                c1 = c1,
                c2 = c2,
                c3 = c3,
                N = N)

        for n,_ in enumerate(self.population):
            self.population[n].specie = self.species[n]

    def prune(self,threshold):
        print(f"[PRUNE]")
        self.population = [genome for genome in self.population if threshold < genome.fitness]

    def evaluate(self):
        ''' function for evaluating genomes into ff networks '''
        networks = []
        for genome in self.population:
            networks.append(compiler(genome,self.inputs))
        return networks

    def update(self,fitness):
        ''' Function for updating fitness '''
        for n,_ in enumerate(self.population):
            self.population[n].fitness = fitness[n]

    def get_params(self):
        params = []
        for n,genome in enumerate(self.population):
            params.append({
                "net" : n,
                "specienumber" : genome.specie,
                "fitness" : genome.fitness,
                "connections" : len(genome.con_gen[genome.con_gen[:,genome.enabled] != 0]),
                "nodes" : len(genome.node_gen[genome.node_gen[:,0] != 0]),
                "nmc" : self.nmc,
                "cmc" : self.cmc,
                "wmc" : self.wmc,
                "bmc" : self.bmc,
                "amc" : self.amc,
                "C1" : self.C1,
                "C2" : self.C2,
                "C3" : self.C3,
                "N" : self.N,
            })
        return params