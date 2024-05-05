from neat import  NodeTypes, FeedForward
import jax.numpy as jnp

LAST_LAYER = 0xDEADBEEF
class Neuron:

    def __init__(self,index , bias, act, type):
        self.index = index
        self.in_layer = self.index
        self.layer = 0
        self.type = type
        self.bias = bias
        self.act  = act
        self.input_list = []
        self.input_neurons = []
        self.weights = []
        if self.type == NodeTypes.INPUT.value:
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

if __name__=="__main__":

    INPUT_SIZE = 3
    OUTPUT_SIZE= 2
    
    neurons = [
        Neuron(0,0.5,0, NodeTypes.INPUT.value),
        Neuron(1,0.5,0, NodeTypes.INPUT.value),
        Neuron(2,0.5,0, NodeTypes.INPUT.value),
        Neuron(3,0.5,0, NodeTypes.NODE.value),
        Neuron(4,0.5,0, NodeTypes.NODE.value),
        Neuron(5,0.5,0, NodeTypes.NODE.value),
        Neuron(6,0.5,0, NodeTypes.NODE.value),
        Neuron(7,0.5,0, NodeTypes.OUTPUT.value),
        Neuron(8,0.5,0, NodeTypes.OUTPUT.value)
    ]

    neurons[3].add_input(neurons[0],1.0)
    neurons[3].add_input(neurons[1],1.0)
    neurons[4].add_input(neurons[2],1.0)
    neurons[5].add_input(neurons[3],1.0)
    neurons[6].add_input(neurons[0],1.0)
    neurons[6].add_input(neurons[3],1.0)
    neurons[6].add_input(neurons[4],1.0)
    neurons[7].add_input(neurons[5],1.0)
    neurons[7].add_input(neurons[0],1.0)
    neurons[8].add_input(neurons[6],1.0)

    FF = FeedForward(INPUT_SIZE,OUTPUT_SIZE)
    FF.add_neurons(neurons)