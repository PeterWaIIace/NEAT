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

from neat import FeedForward, NodeTypes, Genome, Neuron as GNeuron
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
            self.input_list = []
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


def compile_gen2graph(genome):
    ''' compile your network into FF network '''
    # I need to make sure that all output neurons are at the same layer
    ngenomes, cgenomes = genome.node_gen, genome.con_gen
    neurons = []
    active_nodes = ngenomes[ngenomes[:,0] != 0.0]
    for _,node in enumerate(active_nodes):
        neurons.append(
            GNeuron(node)
        )

    for c in cgenomes[cgenomes[:,Genome.enabled] != 0.0]:
        if int(c[Genome.o])-1 < len(neurons) and int(c[Genome.i])-1 < len(neurons):
            neurons[int(c[Genome.o])-1].add_input(
                neurons[int(c[Genome.i])-1],
                c[Genome.w])
            
    return neurons

def test_ff_neat_1():
    INPUT_SIZE = 3
    OUTPUT_SIZE = 2
    payload = jnp.array([1,1,1])
    expected = jnp.array([4.,6.])

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

    result = FF.activate(payload)
    print(expected,result)
    assert jnp.array_equal(expected,result)

def test_ff_neat_2():
    INPUT_SIZE = 3
    OUTPUT_SIZE = 2
    payload = jnp.array([1,1,1])
    expected = jnp.array([4.,7.5])

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
    neurons[6].add_input(neurons[0],1.0)
    neurons[6].add_input(neurons[3],1.0)
    neurons[6].add_input(neurons[4],1.0)

    FF = FeedForward(INPUT_SIZE,OUTPUT_SIZE)
    FF.add_neurons(neurons)
    FF.compile()
    result = FF.activate(payload)

    print(expected,result)
    assert jnp.array_equal(expected,result)

def test_ff_neat_3():
    INPUT_SIZE = 3
    OUTPUT_SIZE = 2
    payload = jnp.array([1,1,1])
    expected = jnp.array([4.5,8.0])

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
    neurons[8].add_input(neurons[6],1.0)

    FF = FeedForward(INPUT_SIZE,OUTPUT_SIZE)
    FF.add_neurons(neurons)
    FF.compile()
    result = FF.activate(payload)

    print(expected,result)
    assert jnp.array_equal(expected,result)

def test_ff_neat_4():
    INPUT_SIZE = 3
    OUTPUT_SIZE = 2
    payload = jnp.array([1,1,1])
    expected = jnp.array([6.0,8.0])

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
    FF.compile()
    result = FF.activate(payload)

    print(expected,result)
    assert jnp.array_equal(expected,result)

NUMBER_OF_ACTIATION_FUNCTIONS = 2
def test_genome_1():
    innov = 0
    genome = Genome()
        
    index =  1
    genome.add_node(index,NodeTypes.INPUT,0.0,0)
    index += 1
    
    genome.add_node(index,NodeTypes.OUTPUT,0.0,0)
    index += 1

    innov = 0
    innov = genome.add_connection(innov,0,1,0.0)

    print(innov)
    assert innov == 1

def test_genome_2():
    innov = 0
    for _ in range(20):
        genome = Genome()
            
        index =  1
        genome.add_node(index,NodeTypes.INPUT,0.0,0)
        index += 1
        
        genome.add_node(index,NodeTypes.OUTPUT,0.0,0)
        index += 1

        innov = 0
        innov = genome.add_r_connection(innov)

        assert innov == 1

def test_genome_3():    
    innov = 0
    for _ in range(20):
        genome = Genome()
            
        index =  1
        genome.add_node(index,NodeTypes.INPUT,0.0,0)
        
        index += 1
        genome.add_node(index,NodeTypes.OUTPUT,0.0,0)

        innov = 0
        innov = genome.add_r_connection(innov)
        
        innov = genome.add_r_node(innov)
        innov_to_check = innov

        innov = genome.add_r_connection(innov)

        assert innov == (innov_to_check + 1)

def test_genome_compilation():
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    genome = Genome()
        
    index =  0
    genome.add_node(index,NodeTypes.INPUT,0.0,0)
    
    index += 1
    genome.add_node(index,NodeTypes.OUTPUT,0.0,0)

    innov = 0
    print("here before")
    innov = genome.add_r_connection(innov)
    
    print("here after")
    innov = genome.add_r_node(innov)
    innov = genome.add_r_connection(innov)
    # innov = genome.add_r_connection(innov)
    # innov = genome.add_r_connection(innov)
    # innov = genome.add_r_connection(innov)
    
    neurons = compile_gen2graph(genome)
    print([neuron.index for neuron in neurons])

    FF = FeedForward(INPUT_SIZE,OUTPUT_SIZE)
    FF.add_neurons(neurons)

    # assert innov == (innov_to_check + 1)

if __name__=="__main__":
    # test_ff_neat_1()
    # test_ff_neat_2()
    # test_ff_neat_3()
    # test_ff_neat_4()
    # test_genome_1()
    # test_genome_2()
    # test_genome_3()
    test_genome_compilation()