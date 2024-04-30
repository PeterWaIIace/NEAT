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
        self.weights = None
        self.index = index
        self.width = max_width
        self.input_size = input_size
        self.residual_connection = jnp.identity(input_size)
        self.neurons = []

    def update_size(self,input_size):
        self.input_size = input_size
        self.width = input_size + len(self.neurons)
        self.residual_connection = jnp.identity(self.input_size)
        
    def add_neuron(self,neuron):
        neuron.in_layer = self.input_size+len(self.neurons)
        self.neurons.append(neuron)
        # what to do if there is more neurons than max length 

    def compile(self,last=False):
        if last == True:
            print("=========== LAST ===========")
            self.width = self.input_size
            weights = jnp.zeros((self.width,len(self.neurons)))
            self.bias = jnp.zeros(len(self.neurons))
        else:
            tmp_weight = jnp.zeros((self.input_size,len(self.neurons)))
            weights    = jnp.concatenate((self.residual_connection, tmp_weight), axis=1).T
            self.bias  = jnp.zeros((self.width))
            
        for n,neuron in enumerate(self.neurons):
            if len(neuron.input_neurons) > 0:
                column = jnp.zeros((self.width))
                inputs = jnp.array([in_neuron.in_layer for in_neuron in neuron.input_neurons],dtype=jnp.int32)
                n_weights = jnp.array(neuron.weights)
                
                column = column.at[inputs].set(n_weights)
                weights = weights.at[:,n].set(column)
            self.bias = self.bias.at[n].set(neuron.bias)
        
        self.weights = weights
        if last == True:
            self.weights = self.weights.T
        return weights.shape[0]
    
class FeedForward:

    def __init__(self,input_size,output_size):
        self.INPUT_SIZE = input_size
        self.max_width = input_size

        self.layers = []
        self.layers.append(Layer(0,self.INPUT_SIZE,self.INPUT_SIZE))

    def add_neurons(self,neurons):
        self.max_width = self.INPUT_SIZE
        for neuron in sorted(neurons,key=lambda neuron: neuron.layer):
            if len(self.layers) > neuron.layer:
                self.layers[neuron.layer].add_neuron(neuron)
            else:
                self.layers.append(Layer(neuron.layer,self.INPUT_SIZE,self.max_width))
                self.layers[neuron.layer].add_neuron(neuron)

            if len(self.layers[neuron.layer].neurons) > self.max_width:
                self.max_width = len(self.layers[neuron.layer].neurons) 

    def compile(self):
        new_input = self.INPUT_SIZE
        for layer in self.layers:
            layer.width = self.max_width
            if layer == self.layers[-1]:
                layer.update_size(new_input)
                new_input = layer.compile(last=True)
            else:
                layer.update_size(new_input)
                new_input = layer.compile()

    def activate(self,x):
        output_values = x
        for layer in self.layers:
            output_values = jnp.dot(output_values,layer.weights.T) + layer.bias
        return output_values

def main():
    INPUT_SIZE = 3
    OUTPUT_SIZE = 2

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

    neurons[3].add_input(neurons[0],1.0)
    neurons[3].add_input(neurons[1],1.0)
    neurons[4].add_input(neurons[3],1.0)
    neurons[5].add_input(neurons[3],1.0)
    neurons[5].add_input(neurons[2],1.0)
    neurons[6].add_input(neurons[4],1.0)
    neurons[6].add_input(neurons[5],1.0)
    neurons[7].add_input(neurons[5],1.0)
    neurons[7].add_input(neurons[0],1.0)

    FF = FeedForward(INPUT_SIZE,OUTPUT_SIZE)
    FF.add_neurons(neurons)
    FF.compile()

    for n in range(20):
        start_time = time.time()
        # input_values = jnp.array([1.0,1.0,1.0])
        input_values = jnp.array([random.random(),random.random(),random.random()])
        output_values = FF.activate(input_values)

        elapsed_time = time.time() - start_time
        if elapsed_time < 0.01:
            print(f"Output val: {colors['purple']}{output_values}{colors['reset']}, elapsed time: {colors['green']}{elapsed_time}{colors['reset']}")
        else:
            print(f"Output val: {colors['purple']}{output_values}{colors['reset']}, elapsed time: {colors['yellow']}{elapsed_time}{colors['reset']}")


if __name__=="__main__":
    main()