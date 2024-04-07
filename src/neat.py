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

from utils import NodeTypes
from evo import NodeGenome, ConnectionGenome 


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

    ff = FeedForward()
    for neuron in neurons:
        ff.add_neuron(neuron)
        
    ff.compile()
    return ff

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

    def __init__(self,index):
        self.layer_index = index
        self.neurons = []    

        self.weigths = None
        self.inputs  = None
        self.outputs = None

        self.inputs_len  = 0
        self.outputs_len = 0
        pass

    def add_neuron(self,neuron):
        self.outputs_len += 1
        if len(neuron.input_list) > 0:
            self.inputs_len = jnp.max(jnp.array(neuron.input_list))+1
        self.neurons.append(neuron)
        # neuron.

    def compile(self):

        self.weigths = jnp.zeros((self.outputs_len,self.inputs_len))
        self.inputs  = jnp.zeros((self.inputs_len),dtype = jnp.int32)
        self.outputs = jnp.zeros((self.outputs_len),dtype = jnp.int32)

        filled_in_length = 0
        for n_i,n in enumerate(self.neurons):
            if len(self.inputs) > 0:
                self.inputs = self.inputs.at[jnp.array(n.input_list)].set(jnp.array(n.input_list))
                filled_in_length += len(n.input_list) 

                self.outputs = self.outputs.at[n_i].set(n.index)

                self.weigths = self.weigths.at[n_i,n.input_list].set(jnp.array(n.weights))            
        # now layer is complied

    def forward(self,input):
        return jnp.dot(input[self.inputs],self.weigths.T)

class FeedForward:

    def __init__(self):
        self.index = 0
        self.size = 0
        self.layers = [Layer(self.index)]

    def add_neuron(self,neuron):
        self.size += 1
        layer_index = neuron.getLayer()
        
        while layer_index >= len(self.layers):
            print(len(self.layers), layer_index)
            self.index += 1
            self.layers.append(Layer(self.index))

        self.layers[layer_index].add_neuron(neuron)
        
    def compile(self):

        for l in self.layers:
            l.compile()

    def activate(self,x):

        output = jnp.zeros(self.size)
        output = output.at[:len(x)].set(x)
                        
        for l in self.layers:
            if l.layer_index > 0:
                output = output.at[l.outputs].set(l.forward(output))

        return output[self.layers[-1].outputs]

    def print(self):

        for l in self.layers:
            print("====================================")
            print(f"weigths: {l.weigths}")
            print(f"inputs:  {l.inputs}")
            print(f"outputs: {l.outputs}")


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
    #### TEST 1 ######
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

    network.print()

    x = jnp.array([1.,1.,1.]) # output should be 1.25 for three times 1. for this network
    out = network.activate(x)
    print(f"result: {out}")
    assert out == 1.25

    #### TEST 2 ######
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
        ConnectionGenome(0,3,5, 0.5, 1),
        ConnectionGenome(0,4,5, 0.5, 1)
    ]

    network = compiler(genome_nodes,genome_connections)

    x = jnp.array([1.,1.,1.]) # output should be 1.25 for three times 1. for this network
    out = network.activate(x)
    print(f"result: {out}")
    assert out == 1


