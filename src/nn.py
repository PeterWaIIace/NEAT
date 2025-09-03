
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
from src.misc import δ, sh, speciate, mate, cross_over
# First networkx library is imported
# along with matplotlib

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
