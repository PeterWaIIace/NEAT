

## 1) So what this neat has to do, it has to accept genome and decode it into network
## 2) Each genome is divided into cells containing node parameters [connection, bias ]

# The idea is that different representations are appro-priate for different kinds of operators.
# Subgraph-swapping crossovers and topological
# mutations use the grid, while point crossovers and connection parameter mutations
# use the linear representation

# NEAT performs artificial synapsis based on historical markings, allowing it to add new
# structure without losing track of which gene is which over the course of a simulation

# Genome
# Innovation_index
# Node
# Connection
# In
# Out
# Weight
import jax.numpy as jnp
from enum import Enum

class NodeTypes(Enum):
    HIDDEN = 1 
    INPUT = 2
    OUTPUT = 3 

class NodeGenome:

    def __init__(self,type,index):
        self.type  = type 
        self.index = index 

class ConnectionGenome:

    def __init__(self,innov,in_index,out_index,weight,enabled):
        self.innov = innov 
        self.in_neuron = in_index 
        self.out_neuron = out_index
        self.weight = weight
        self.enabled = enabled 

def compiler(ngenomes : [NodeGenome], cgenomes : [ConnectionGenome]):

    neurons = []
    for n in ngenomes:
        neurons.append(
            Neuron(n)              
        )

    for c in cgenomes:
        if c.enabled:
            neurons[c.out_neuron].add_input(
                c.in_neuron,
                c.weight,
                neurons[c.in_neuron].layer)

    return neurons

class Neuron:

    def __init__(self,node_genome):
        self.index = node_genome.index
        self.layer = 0
        self.input_list = []
        self.weights = []

    def add_input(self,in_neuron,weigth,layer):
        self.input_list.append(in_neuron)
        self.weights.append(weigth)
        if layer == self.layer:
            self.layer+=1
        if layer > self.layer:
            self.layer = layer + 1
        
    def print(self):
        print(
        f"self.index:     {self.index}\
          self.layer:     {self.layer}\
          self.input_list {self.input_list}\
          self.weights    {self.weights}"
        )

class Layer:

    def __init__(self):
        self.neurons = []
        pass

    def add_neuron(self,neuron):
        self.neurons.append(neuron)

    def compile(self):
        self.weigths = jnp.array([neuron.weights for neuron in self.neurons])
        self.inputs = jnp.zeros()


class FeedForward:

    def __init__(self,layers):
                

class Neat:

    def __init__(self):
        pass


class Network:

    def __init__(self): 
        self.genome = []
        pass

    def evalueate(self,genome):
        self.genom = genome
        

        pass

if __name__=="__main__":
    pass

    genome_nodes = [
        NodeGenome(NodeTypes.INPUT,0),
        NodeGenome(NodeTypes.INPUT,1),
        NodeGenome(NodeTypes.INPUT,2),
        NodeGenome(NodeTypes.HIDDEN,3),
        NodeGenome(NodeTypes.HIDDEN,4),
        NodeGenome(NodeTypes.OUTPUT,5)
    ]

    genome_connections = [
        ConnectionGenome(0,0,3, 0.5, 1),
        ConnectionGenome(0,1,3, 0.5, 1),
        ConnectionGenome(0,1,4, 0.5, 1),
        ConnectionGenome(0,2,4, 0.5, 1),
        ConnectionGenome(0,3,4, 0.5, 1),
        ConnectionGenome(0,4,5, 0.5, 1),
        ConnectionGenome(0,1,5, 0.5, 1)
    ]

    network = compiler(genome_nodes,genome_connections)

    for n in network:
        n.print()