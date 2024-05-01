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

from neat import FeedForward, NodeTypes
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

if __name__=="__main__":
    
    INPUT_SIZE = 3
    OUTPUT_SIZE = 2

    input_values = jnp.array([1.0,1.0,1.0])

    layers = []
    neurons = [
        Neuron(0,0.5,0, NodeTypes.INPUT.value),
        Neuron(1,0.5,0, NodeTypes.INPUT.value),
        Neuron(2,0.5,0, NodeTypes.INPUT.value),
        Neuron(3,0.5,0, NodeTypes.NODE.value),
        Neuron(4,0.5,0, NodeTypes.NODE.value),
        Neuron(5,0.5,0, NodeTypes.OUTPUT.value),
        Neuron(6,0.5,0, NodeTypes.OUTPUT.value)
    ]

    neurons[3].add_input(neurons[0],1.0)
    neurons[3].add_input(neurons[1],1.0)
    neurons[4].add_input(neurons[2],1.0)
    neurons[5].add_input(neurons[3],1.0)
    neurons[6].add_input(neurons[3],1.0)
    neurons[6].add_input(neurons[4],1.0)

    FF = FeedForward(INPUT_SIZE,OUTPUT_SIZE)
    FF.add_neurons(neurons)
    FF.compile()

    payload = jnp.array([1,1,1])
    expected = jnp.array([4.,6.])
    result = FF.activate(payload)
    print(expected,result)
    assert jnp.array_equal(expected,result)