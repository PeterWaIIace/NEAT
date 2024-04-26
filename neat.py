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

from enum import Enum

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

    def uniform(self,max=1.0,min=0):
        random_float = jrnd.uniform(self.key, shape=(1,), minval=min, maxval=max)
        self.key = jrnd.split(self.key,1)[0]
        return random_float[0]
        

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

def activation_func(x,y):
    ''' branchless and vectorized activation functions'''
    return jnp.where(y == 0, x,
            jnp.where(y == 1, sigmoid(x),
            jnp.where(y == 2, ReLU(x),
            jnp.where(y == 3, leakyReLU(x),
            jnp.where(y == 4, softplus(x),
                        tanh(x))))))
class Genome:

    i = 2
    o = 3
    w = 4
    enabled = 5

    n_index = 0
    n_bias = 2
    n_act  = 3

    def __init__(self):
        self.connections_length = 20
        self.nodes_length = 20
        self.max_innov = 0
        self.index = 0
        self.fitness = 1.0
        self.specie = 0
        # helper indicies names
        # Connections genomes is array, rows are genomes, but cols are parameters of that genomes
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
            self.nodes_length += 20
            new_connections_spaces = jnp.zeros((20,6),)
            self.con_gen = jnp.concatenate((self.con_gen,new_connections_spaces), axis=0)

        if self.node_gen[int(in_node),self.n_index] == 0:
            self.add_node(in_node,NodeTypes.NODE,0.0,Rnd.randint(max=NUMBER_OF_ACTIATION_FUNCTIONS,min=0))

        if self.node_gen[int(out_node),self.n_index] == 0:
            self.add_node(out_node,NodeTypes.NODE,0.0,Rnd.randint(max=NUMBER_OF_ACTIATION_FUNCTIONS,min=0))

        self.con_gen = self.con_gen.at[innov].set(jnp.array([innov,innov,in_node+1,out_node+1,weight,1.0]))
        return innov

    def change_weigth(self,weigth):
        exisitng_connections = self.con_gen[self.con_gen[:,0] != 0]
        index_of_connection = Rnd.randint(max=len(exisitng_connections))
        self.con_gen = self.con_gen.at[index_of_connection,self.w].set(weigth)

    def change_bias(self,bias):
        existing_nodes = self.node_gen[self.node_gen[:,0] != 0]
        node_index = Rnd.randint(max=len(existing_nodes))
        self.node_gen = self.node_gen.at[node_index,self.n_bias].set(bias)

    def change_activation(self,act):
        existing_nodes = self.node_gen[self.node_gen[:,0] != 0]
        node_index = Rnd.randint(max=len(existing_nodes))
        self.node_gen = self.node_gen.at[int(node_index),self.n_act].set(act)


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
def cross_over(population : list,keep_top : int = 2, δ_th : float = 5, **kwargs):
    ''' cross over your population '''
    keep_top = int(keep_top)
    if keep_top < 2:
        keep_top = 2

    new_population = []
    species = speciate(population, δ_th, **kwargs)
    species_list = []

    for s_n,specie in enumerate(species):
        sorted_specie = sorted(specie, key=lambda x: x.fitness, reverse=True)[:]
        top_species = sorted_specie[:keep_top]
        for keept in top_species:
            new_population.append(keept)
            species_list.append(s_n)

        for _ in range(len(sorted_specie) - keep_top):
            n = random.randint(0,len(top_species)-1)
            m = n
            while m == n:
                m = random.randint(0,len(top_species)-1)
            offspring = mate(top_species[n],top_species[m])
            new_population.append(offspring)
            species_list.append(s_n)
    
    population = []
    return new_population, species_list

def random_mutate(population,
                innov = 0,
                nmc = 0.7, 
                cmc = 0.7,
                wmc = 0.7,
                amc = 0.7,
                bmc = 0.7):
    for n,_ in enumerate(population):

        mutation_chance = random.randint(0,10)
        if mutation_chance > nmc*10:
            innov = population[n].add_r_node(innov)

        mutation_chance = random.randint(0,10)
        if mutation_chance  > cmc*10:
            innov = population[n].add_r_connection(innov)

        mutation_chance = random.randint(0,10)
        if mutation_chance  > wmc*10:
            population[n].change_weigth(Rnd.uniform())

        mutation_chance = random.randint(0,10)
        if mutation_chance  > amc*10:
            population[n].change_activation(Rnd.randint(NUMBER_OF_ACTIATION_FUNCTIONS,0))

        mutation_chance = random.randint(0,10)
        if mutation_chance  > bmc*10:
            population[n].change_bias(Rnd.uniform())

    return innov

def evolve(population,innov,mutate_rate,**kwargs):
    population, species_list = cross_over(population,
                        keep_top = kwargs.get('keep_top',2),
                        δ_th = kwargs.get('δ_th',5),
                        c1 = kwargs.get('C1',1),
                        c2 = kwargs.get('C2',1),
                        c3 = kwargs.get('C3',1),
                        N = kwargs.get('N',1))

    for _ in range(mutate_rate):
        innov = random_mutate(population,
                            innov,
                            nmc = kwargs.get('nmc',0.7),
                            cmc = kwargs.get('cmc',0.7),
                            wmc = kwargs.get('wmc',0.7),
                            amc = kwargs.get('amc',0.7),
                            bmc = kwargs.get('bmc',0.7))
    return (population, innov, species_list)

def compiler(ngenomes, cgenomes):
    # I need to make sure that all output neurons are at the same layer
    neurons = []
    active_nodes = ngenomes[ngenomes[:,0] != 0.0]
    for n,node in enumerate(active_nodes):
        neurons.append(
            Neuron(node)
        )
        # neurons[n].inputs  = cgenomes[cgenomes[:,Genome.o] == n][:,Genome.i]
        # neurons[n].weights = cgenomes[cgenomes[:,Genome.o] == n][:,Genome.i]

    for c in cgenomes[cgenomes[:,Genome.enabled] != 0.0]:
        if Genome.o < len(c):
            if int(c[Genome.o])-1 < len(neurons) and int(c[Genome.i])-1 < len(neurons):
                try:
                    neurons[int(c[Genome.o])-1].add_input(
                        int(c[Genome.i])-1,
                        c[Genome.w],
                        neurons[int(c[Genome.i])-1].layer)
                except Exception as e:
                    print(f"caught: {e}")
                    print(int(c[Genome.i])-1,int(c[Genome.o])-1, len(neurons),neurons)
                    quit(-1)
                
    ff = FeedForward(ngenomes, cgenomes)
    for neuron in neurons:
        ff.add_neuron(neuron)

    ff.compile()
    return ff

class Neuron:

    def __init__(self,node_genome):
        self.index = int(node_genome[0]) - 1
        self.layer = 0
        self.type = int(node_genome[1])
        self.bias = node_genome[Genome.n_bias]
        self.act  = node_genome[Genome.n_act]
        self.input_list = []
        self.weights = []

        
    def add_input(self,in_neuron,weigth,layer):
        self.input_list.append(in_neuron)
        self.weights.append(weigth)
        if layer >= self.layer:
            self.layer = layer + 1

    def getLayer(self):
        return self.layer

    def get(self):
        return {self.layer : jnp.array(self.weights)}

    def print(self):
        print(
        f"self.index:     {self.index}\
          self.layer:     {self.layer}\
          self.input_list {self.input_list}\
          self.weights    {self.weights}"
        )

class Layer:
    # TODO: distinguish between output layers and rest

    def __init__(self,index):
        self.layer_index = index
        self.neurons = []

        self.weigths = None
        self.inputs  = None
        self.outputs = None
        self.biases = None
        self.acts = None

        self.inputs_len  = 0
        self.outputs_len = 0

        self.vmap_activate = jax.vmap(activation_func)

    def add_neuron(self,neuron):
        self.outputs_len += 1
        if len(neuron.input_list) > 0:
            self.inputs_len = jnp.max(jnp.array(neuron.input_list))+1
        self.neurons.append(neuron)
        # neuron.

    def compile(self):

        self.weigths = jnp.zeros((self.outputs_len,self.inputs_len),dtype = jnp.float32)
        self.inputs  = jnp.zeros((self.inputs_len),dtype = jnp.int32)
        self.outputs = jnp.zeros((self.outputs_len),dtype = jnp.int32)
        self.acts   = jnp.zeros((self.outputs_len),dtype = jnp.int32)
        self.biases = jnp.zeros((self.outputs_len),dtype = jnp.float32)

        filled_in_length = 0
        for n_i,n in enumerate(self.neurons):
            if len(self.inputs) > 0:
                self.inputs = self.inputs.at[jnp.array(n.input_list)].set(jnp.array(n.input_list))
                filled_in_length += len(n.input_list)

                self.outputs = self.outputs.at[n_i].set(n.index)
                self.biases = self.biases.at[n_i].set(n.bias)
                self.acts = self.acts.at[n_i].set(n.act)

                self.weigths = self.weigths.at[n_i,n.input_list].set(jnp.array(n.weights))  
        # now layer is complied

    def forward(self,input):
        X = jnp.dot(input[self.inputs],self.weigths.T) + self.biases
        return activation_func(X,self.acts)

class FeedForward:

    def __init__(self,ngenome,cgenome):
        self.ngenome = ngenome
        self.cgenome = cgenome
        self.index = 0
        self.size = 0
        self.layers = [Layer(self.index)]
        self.output_layer = Layer(999)
        self.graph = nx.DiGraph()

    def dump_genomes(self):
        return {"nodes":self.ngenome,"connect" : self.cgenome}
    
    def add_neuron(self,neuron):
        self.size += 1
        if neuron.type == NodeTypes.OUTPUT.value:
            self.output_layer.add_neuron(neuron)
        else:
            layer_index = neuron.getLayer()

            while layer_index >= len(self.layers):
                self.index += 1
                self.layers.append(Layer(self.index))

            self.layers[layer_index].add_neuron(neuron)

    def compile(self):
        self.index += 1
        self.output_layer.index = self.index
        self.layers.append(self.output_layer)

        for l in self.layers:
            l.compile()

    def activate(self,x):
        output = jnp.zeros(self.size)
        output = output.at[:len(x)].set(x)

        for l in self.layers[1:]:
            output = output.at[l.outputs].set(l.forward(output))

        return output[self.layers[-1].outputs]

    def print(self):

        for l in self.layers:
            print("====================================")
            print(f"weigths: {l.weigths}")
            print(f"inputs:  {l.inputs}")
            print(f"outputs: {l.outputs}")

    def __add_edge_to_graph(self,n1,n2,label,thickness):
        # if n1 in self.graph.nodes() and n2 in self.graph.nodes():
        self.graph.add_edge(n1,n2,label=label,thickness=thickness)

    def __add_node_to_graph(self,node,x,max_layer = 10):

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
            nx.set_node_attributes(self.graph, {node.index: (x * 5,max_layer+1)}, "pos")
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
                    self.__add_node_to_graph(neuron,x,max_layer = len(self.layers))
            
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
        except:
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
                self.population[n].add_node(index,NodeTypes.INPUT,Rnd.uniform(min=-1.0,max=1.0),Rnd.randint(max=NUMBER_OF_ACTIATION_FUNCTIONS))
                index += 1
            
            output_offset = index
            for o in range(self.outputs):
                self.population[n].add_node(index,NodeTypes.OUTPUT,Rnd.uniform(min=-1.0,max=1.0),Rnd.randint(max=NUMBER_OF_ACTIATION_FUNCTIONS))
                index += 1

            creation_innov = 0
            for i in range(self.inputs):
                for o in range(self.outputs):
                    creation_innov = self.population[n].add_connection(creation_innov,i,output_offset+o,Rnd.uniform(min=-1.0,max=1.0))
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

    def evolve(self,mutate_rate = 2):
        self.population,self.innov,self.species = evolve(self.population,
                                            self.innov,
                                            mutate_rate = mutate_rate,
                                            nmc = self.nmc,
                                            cmc = self.cmc,
                                            wmc = self.wmc,
                                            bmc = self.bmc,
                                            amc = self.amc,
                                            C1 = self.C1,
                                            C2 = self.C2,
                                            C3 = self.C3,
                                            N = self.N,
                                            δ_th = self.δ_th)

        for n,_ in enumerate(self.population):
            self.population[n].specie = self.species[n]

    def evaluate(self):
        ''' function for evaluating genomes into ff networks '''
        networks = []
        for genome in self.population:
            networks.append(compiler(genome.node_gen,genome.con_gen))
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

# def run():
#     # env = gym.make("Acrobot-v1", render_mode="human")
#     # env = gym.make("Acrobot-v1")
#     my_neat = NEAT(6,3, 20)

#     epochs = 50
#     prev_action = 0.0
#     experiment_length = 100
#     models_path = "models"
#     game = "SlimeVolley-v0"
#     for e in range(epochs):
#         print(f"================ EPOCH: {e} ================")
#         env = gym.make(game)
#         observation, info = env.reset(seed=42)
#         all_rewards = []
#         my_neat.evolve()

#         networks = my_neat.evaluate()
#         for n,network in enumerate(networks):
#             observation, info = env.reset()
#             total_reward = 0

#             for _ in range(experiment_length):
#                 actions = network.activate(jnp.array(observation))
#                 action = actions.argmax()
#                 #promote mobility
#                 if prev_action != action:
#                     total_reward += abs(observation[4])/10000 + abs(observation[5])/10000
#                     prev_action = action

#                 observation, reward, terminated, truncated, info = env.step(action)
#                 total_reward += reward
#                 if terminated or truncated:
#                     break

#             all_rewards.append(total_reward)
#             print(f"net: {n}, fitness: {total_reward}")

#         env.close()

#         #display the best:
#         index = all_rewards.index(max(all_rewards))
#         network = networks[index]
#         env = gym.make(game, render_mode="human")

#         print(f"Displaying the best: {index}, max reward: {max(all_rewards)}")
#         observation, info = env.reset()
#         total_reward = 0

#         for _ in range(experiment_length):
#             actions = network.activate(jnp.array(observation))
#             action = actions.argmax()
#             #promote mobility
#             if prev_action != action:
#                 total_reward += abs(observation[4])/10000 + abs(observation[5])/10000
#                 prev_action = action

#             observation, reward, terminated, truncated, info = env.step(action)
#             total_reward += reward
#             if terminated or truncated:
#                 break

#         pickle.dump(network,open(f"{models_path}/{game}_e{e}.neatpy","wb"))
#         my_neat.update(all_rewards)

#         env.close()

# if __name__=="__main__":
#     run()

    # population = [superior, inferior]
    # print(speciate(population))

    # print(inferior.con_gen)
    # print(inferior.node_gen)
    # inferior = mate(superior,inferior)
    # print(inferior.con_gen)
    # print(inferior.node_gen)