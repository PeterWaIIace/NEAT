import random
import networkx as nx
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.random import StatefulRandomGenerator
from enum import Enum 
from arrayPainter import display_array, display_with_values

Rnd = StatefulRandomGenerator()

class NodeTypes(Enum):
    NODE   = 1
    INPUT  = 2
    OUTPUT = 3

class Genome:

    CHUNK = 20

    CONNECTION_SIZE = 5
    C_INNOV = 0
    C_IN  = 1
    C_OUT = 2
    C_W   = 3
    enabled = 4

    NODE_SIZE = 5
    N_INDEX = 0
    N_TYPE = 1
    N_BIAS = 2
    N_ACT  = 3
    N_EN   = 4

    def __init__(self):
        self.index = 0
        self.specie = 0
        self.fitness = 1.0
        self.max_innov = 0

        self.connections = jnp.zeros((self.CHUNK,self.CONNECTION_SIZE),)
        self.nodes = jnp.zeros((self.CHUNK,self.NODE_SIZE),)

    def __conn_exists(self,in_node,out_node):

        forward_connection = ((self.connections[:,self.C_IN] == in_node) * (self.connections[:,self.C_IN] == out_node)).any()  
        reverse_connection = ((self.connections[:,self.C_IN] == out_node) * (self.connections[:,self.C_IN] == in_node)).any()  

        return forward_connection or reverse_connection

    def __get_possible_conn(self):
        active_nodes = self.nodes[self.nodes[:,self.N_EN] != 0]
        input_nodes  = active_nodes[active_nodes[:,self.N_TYPE] != float(NodeTypes.OUTPUT.value)][:,self.N_INDEX]
        output_nodes = active_nodes[active_nodes[:,self.N_TYPE] != float(NodeTypes.INPUT.value)][:, self.N_INDEX]
        pairs = []
        for input_node in input_nodes:
            for output_node in output_nodes:
                if not self.__conn_exists(input_node,output_node) and input_node != output_node:
                    pairs.append((input_node,output_node))

        return pairs

    def add_node(self, type : int, bias : float, act : int):
        ''' Adding node '''

        new_node_index = len(self.nodes[self.nodes[:,self.N_EN] != 0.0])

        if len(self.nodes) <= new_node_index:
            self.nodes = jnp.concatenate((self.nodes,jnp.zeros((self.CHUNK,self.CONNECTION_SIZE),)), axis=0)

        enabled = 1
        new_node_values = jnp.array([new_node_index,type,bias,act,enabled])
        self.nodes = self.nodes.at[new_node_index].set(new_node_values)
        return new_node_index

    def add_connection(self, innov : int, in_node, out_node, weight : float):
        ''' Adding connection '''
        if self.nodes[in_node,self.N_EN] == 0 or self.nodes[out_node,self.N_EN] == 0 or self.__conn_exists(in_node, out_node):
            return innov
        
        if len(self.connections) <= innov:
            self.connections = jnp.concatenate((self.connections,jnp.zeros((self.CHUNK,self.CONNECTION_SIZE),)), axis=0)

        new_connection_values = jnp.array([innov,in_node,out_node,weight,1.0])
        self.connections= self.connections.at[innov].set(new_connection_values)
        
        self.max_innov = innov + 1
        return self.max_innov

    def add_r_connection(self,innov):
        
        pairs = self.__get_possible_conn()
        if len(pairs) <= 0:
            return innov
        
        random.shuffle(pairs)
        
        pair = pairs[0]
        in_node  = pair[0]
        out_node = pair[1]
        
        innov = self.add_connection(int(innov),int(in_node),int(out_node),1.0)
        return innov

    def add_r_node(self,innov : int):
        exisitng_connections = self.connections[self.connections[:,self.enabled] != 0]

        chosen_connection = exisitng_connections[Rnd.randint(max=len(exisitng_connections))]
        self.connections  = self.connections.at[int(chosen_connection[self.C_INNOV]),self.enabled].set(0.0)

        new_node = self.add_node(NodeTypes.NODE.value, 0.0, 0)
        in_node  = int(chosen_connection[self.C_IN])

        innov = self.add_connection(innov,
                            in_node,
                            int(new_node),
                            chosen_connection[self.C_W]
                        )

        out_node = int(chosen_connection[self.C_OUT])
        innov = self.add_connection(innov,
                            int(new_node),
                            out_node,
                            chosen_connection[self.C_W]
                        )
        return innov

    def change_weigth(self,weigth):
        self.connections= self.connections.at[:,self.C_W].add(weigth)
        
    def change_bias(self,bias):
        self.nodes = self.nodes.at[:,self.N_BIAS].add(bias)

    def change_activation(self,act):
        self.nodes = self.nodes.at[:,self.N_ACT].set(act)

class Node:

    def __init__(self,index,type,bias,act):
        self.index = index
        self.type = type
        self.bias = bias
        self.act  = act
        self.layer = 0

        self.weights = []
        self.inputs = []
        if self.type == NodeTypes.INPUT.value:
            self.weights = [1.0]

        if self.type == NodeTypes.OUTPUT.value:
            self.layer = LAST_LAYER

    def add_input(self,input_node, input_weight):
        self.weights.append(input_weight)
        self.inputs.append(input_node)
        for node in self.inputs:
            if node.layer >= self.layer:
                self.layer = node.layer + 1

def grapprocessor(genome):
    ''' compile your network into FF network '''
    # I need to make sure that all output neurons are at the same layer
    ngenomes, cgenomes = genome.nodes, genome.connections
    nodes = []
    active_nodes = ngenomes[ngenomes[:,Genome.N_EN] != 0.0]
    for _,node in enumerate(active_nodes):
        nodes.append(
            Node(
                node[Genome.N_INDEX],
                node[Genome.N_TYPE],
                node[Genome.N_BIAS],
                node[Genome.N_ACT]
            )
        )

    # TODO: here is an error
    for c in cgenomes[cgenomes[:,Genome.enabled] != 0.0]:
        nodes[int(c[Genome.C_OUT])].add_input(
            nodes[int(c[Genome.C_IN])],
            c[Genome.C_W]
        )

    # nodes = topologicalSort(nodes)
    return nodes

def visit(visited,node,sorted):
    ''' visit next nodes '''
    visited[int(node.index)] = True

    for n_node in node.inputs:
        if not visited[int(n_node.index)]:
            visit(visited,n_node,sorted)

    sorted.insert(0,node)

def topologicalSort(nodes):
    ''' execute topological sort on nodes '''

    visited = [False] * len(nodes)
    sorted = []

    for node in reversed(nodes):
        if not visited[int(node.index)]:
            visit(visited,node,sorted)
    return sorted

def compiler(nodes,input_size,genome):
    ''' compile your network into FF network '''
    # I need to make sure that all output neurons are at the same layer

    ff = FeedForward(input_size,genome)
    ff.add_neurons(nodes)

    ff.compile()
    return ff

LAST_LAYER = 0xDEADBEEF

class Layer:

    def __init__(self,index,input_size,max_width):
        self.weights = None
        self.index = index
        self.width = max_width
        self.input_size = input_size
        self.neurons = []
        self.neurons_index_offset = 0
        # self.vmap_activate = jax.vmap(activation_func)

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
                weights  = jnp.concatenate((self.residual_connection, tmp_weights), axis=1)
            else:
                weights = tmp_weights
            self.bias = jnp.zeros((self.width))
            self.acts = jnp.zeros((self.width),dtype=jnp.int32)
            
        display_array([weights,self.bias],["blue","green"])

        for n,neuron in enumerate(self.neurons):
            # update all neurons indexes based on offset in this layer
            neuron.in_layer += self.neurons_index_offset

            if len(neuron.inputs) > 0:
                column = jnp.zeros((self.input_size))
                inputs = jnp.array([in_neuron.in_layer for in_neuron in neuron.inputs],dtype=jnp.int32)
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

        display_array([weights,self.bias],["blue","green"])

        return self.bias.shape[0]
    
class FeedForward:

    def __init__(self,input_size,genome):
        self.genome     = genome
        self.INPUT_SIZE = input_size
        self.max_width  = input_size

        self.layers  = []
        self.neurons = []
        self.layers.append(Layer(0,0,0))
        self.graph = nx.DiGraph()

    def dump_genomes(self):
        return {"nodes":self.genome.node_gen,"connect" : self.genome.con_gen}

    def add_neurons(self,neurons):
        self.neurons   = neurons
        self.max_width = self.INPUT_SIZE
        sorted_neurons = sorted(neurons,key=lambda neuron: neuron.layer)

        output_neurons = [neuron for neuron in sorted_neurons if neuron.layer == LAST_LAYER]
        sorted_neurons = [neuron for neuron in sorted_neurons if neuron.layer != LAST_LAYER]

        for neuron in sorted_neurons:
            if len(self.layers) > neuron.layer:
                self.layers[neuron.layer].add_neuron(neuron)
            else:
                for n in range(neuron.layer - (len(self.layers)-1)):
                    self.layers.append(Layer(len(self.layers) + n, self.INPUT_SIZE, self.max_width))

                self.layers[neuron.layer].add_neuron(neuron)

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
        print("==============================================")
        print([(neuron.input_list,neuron.index) for neuron in self.neurons])
        for layer in self.layers:
            display_array([output_values,layer.weights,layer.bias],["blue","red","green"])
            output_values = jnp.dot(output_values,layer.weights) + layer.bias
            output_values = activation_func(output_values,layer.acts)
        return output_values
    
    def dry(self):
        print("----------------------------------------")
        for layer in self.layers:
            display_array([layer.weights,layer.bias],["red","green"])
            print(layer.weights)
            print(layer.bias)


if __name__=="__main__":
    g = Genome()

    for n in range(12):
        g.add_node(NodeTypes.INPUT.value,0.0,0.0)
    for n in range(3):
        g.add_node(NodeTypes.OUTPUT.value,0.0,0.0)
    
    innov = 0
    for n in range(40):
        innov = g.add_r_connection(innov)

    for n in range(4):
        for n in range(5):
            innov = g.add_r_node(innov)

        for n in range(5):
            innov = g.add_r_connection(innov)

    nodes = grapprocessor(g)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    node_color_map = {NodeTypes.INPUT.value: "skyblue", NodeTypes.NODE.value: "lightgreen", NodeTypes.OUTPUT.value: "salmon"}
    print([([int(input.index) for input in node.inputs],int(node.index)) for node in nodes])
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
    pos = nx.spring_layout(G)  # Layout for better visualization
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=12, font_weight="bold")
    # plt.ion()
    plt.show()
    FF = compiler(nodes,12,g)

