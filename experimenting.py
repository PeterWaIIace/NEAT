import gymnasium as gym
import slimevolleygym
import jax.numpy as jnp
import numpy as np
import argparse
import pickle 
import random
import time
import csv
import sys
import os

from neat import NEAT
from enum import Enum

N = 10
GENERATIONS = 20
POPULATION_SIZE = 4
NMC = 1.0
CMC = 1.0
WMC = 1.0
BMC = 1.0
AMC = 1.0
δ_th = 5
MUTATE_RATE = 16
RENDER_HUMAN = True

# Define color codes
colors = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'purple': '\033[95m',
    'orange': '\033[33m',  # Orange color
    'reset': '\033[0m'  # Reset color to default
}


class NodeTypes(Enum):
    NODE   = 1
    INPUT  = 2
    OUTPUT = 3

class Neuron:

    def __init__(self, index : int, bias : float, act : int, type: int):
        self.index = index
        self.in_layer = index
        self.layer = 0
        self.type = type
        self.bias = bias
        self.act  = act
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

        for neuron in self.input_neurons:
            if neuron.layer >= self.layer:
                self.layer = neuron.layer + 1

    def getLayer(self):
        return self.layer

    def get(self):
        return {self.layer : jnp.array(self.weights)}


class Layer:

    def __init__(self,index,input_size,max_width):
        self.index = index
        self.width = max_width
        self.input_size = input_size
        self.neurons = []

    def add_neuron(self,neuron):
        neuron.in_layer = len(self.neurons)
        self.neurons.append(neuron)
        # what to do if there is more neurons than max length 
                
    def compile(self,last=False):
        if self.index != 0:
            self.input_size = self.width

        if last == True:
            self.weights = jnp.zeros((len(self.neurons),self.width))
            self.bias = jnp.zeros((len(self.neurons)))
        else:
            self.weights = jnp.identity(self.width)[:,:self.input_size]
            self.bias = jnp.zeros((self.width))

        for n,neuron in enumerate(self.neurons):
            if len(neuron.input_neurons) > 0:
                column = jnp.zeros((self.width))
                inputs = jnp.array([in_neuron.in_layer for in_neuron in neuron.input_neurons],dtype=jnp.int32)
                weights = jnp.array(neuron.weights)
                column = column.at[inputs].set(weights)
                
                t_column = column.T

                self.weights = self.weights.at[n,:].set(t_column)
            self.bias = self.bias.at[n].set(neuron.bias)


def main():
    INPUT_SIZE = 3

    input_values = jnp.array([1.0,1.0,1.0])

    layers = []
    neurons = [
        Neuron(0,0.5,1, NodeTypes.INPUT.value),
        Neuron(1,0.5,1, NodeTypes.INPUT.value),
        Neuron(2,0.5,1, NodeTypes.INPUT.value),
        Neuron(3,0.5,1, NodeTypes.NODE.value),
        Neuron(4,0.5,1, NodeTypes.NODE.value),
        Neuron(5,0.5,1, NodeTypes.NODE.value),
        Neuron(6,0.5,1, NodeTypes.OUTPUT.value),
        Neuron(7,0.5,1,NodeTypes.OUTPUT.value)
    ]

    layers.append(Layer(0,INPUT_SIZE,INPUT_SIZE))
    
    neurons[3].add_input(neurons[0],1.0)
    neurons[3].add_input(neurons[1],1.0)
    neurons[4].add_input(neurons[3],1.0)
    neurons[5].add_input(neurons[3],1.0)
    neurons[5].add_input(neurons[2],1.0)
    neurons[6].add_input(neurons[4],1.0)
    neurons[6].add_input(neurons[5],1.0)
    neurons[7].add_input(neurons[5],1.0)

    max_width = INPUT_SIZE
    for neuron in sorted(neurons,key=lambda neuron: neuron.layer):
        if len(layers) > neuron.layer:
            layers[neuron.layer].add_neuron(neuron)
        else:
            layers.append(Layer(neuron.layer,INPUT_SIZE,max_width))
            layers[neuron.layer].add_neuron(neuron)

        if len(layers[neuron.layer].neurons) > max_width:
            max_width = len(layers[neuron.layer].neurons) 

    for layer in layers:
        layer.width = max_width
        if layer == layers[-1]:
            layer.compile(last=True)
        else:
            layer.compile()

    for n in range(20):
        start_time = time.time()
        output_values = jnp.array([random.random(),random.random(),random.random()])
        for layer in layers:
            output_values = jnp.dot(output_values,layer.weights.T) + layer.bias

        elapsed_time = time.time() - start_time
        if elapsed_time < 0.01:
            print(f"Output val: {colors['purple']}{output_values}{colors['reset']}, elapsed time: {colors['green']}{elapsed_time}{colors['reset']}")
        else:
            print(f"Output val: {colors['purple']}{output_values}{colors['reset']}, elapsed time: {colors['yellow']}{elapsed_time}{colors['reset']}")


if __name__=="__main__":
    main()