import os
import sys
import jax 
import jax.numpy as jnp

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,f"{dir_path}/../")

from src.evo import Genome
from src.neat import compiler

def test_disable_connection_node_1():

    genome = Genome()

    genome.cgenome[0].enabled = 0
    network = compiler(genome.ngenome,genome.cgenome)

    x = jnp.array([1.,1.,1.]) # output should be 1.25 for three times 1. for this network
    out = network.activate(x)
    print(f"result: {out}")
    assert out == 2

def test_adding_node_1():

    genome = Genome()

    genome.cgenome[0].enabled = 0

    genome.set_innovation(0)
    assert True is genome.add_node(genome.cgenome[0],1.0)
    innovation = genome.get_innovation()

    network = compiler(genome.ngenome,genome.cgenome)

    x = jnp.array([1.,1.,1.]) # output should be 1.25 for three times 1. for this network
    out = network.activate(x)
    print(f"result: {out}")
    assert out == 3
    assert innovation == 2

def test_adding_connection_1():

    genome = Genome()
    for cgenome in genome.cgenome:
        print(cgenome.in_neuron,cgenome.out_neuron)

    genome.set_innovation(0)
    assert True is genome.add_node(genome.cgenome[0],1.0)
    assert True is genome.add_node(genome.cgenome[0],1.0)
    assert True is genome.add_connection(2,genome.ngenome[-1].index,1.0)
    innovation = genome.get_innovation()

    network = compiler(genome.ngenome,genome.cgenome)

    x = jnp.array([1.,1.,1.]) # output should be 1.25 for three times 1. for this network
    out = network.activate(x)
    assert out == 5
    assert innovation == 5

def test_adding_connection_2():

    genome = Genome()

    genome.set_innovation(0)

    genome.add_node(genome.cgenome[0],1.0)
    assert True  is genome.add_connection(2, genome.ngenome[-1].index, 1.0)
    assert False is genome.add_connection(genome.ngenome[-1].index, 2, 1.0)