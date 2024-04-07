import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,f"{dir_path}/../")

from src.neat import *


def test_compilation_of_network_1():
    genome_nodes = [
        NodeGene(NodeTypes.INPUT,0),
        NodeGene(NodeTypes.INPUT,1),
        NodeGene(NodeTypes.INPUT,2),
        NodeGene(NodeTypes.HIDDEN,3),
        NodeGene(NodeTypes.HIDDEN,4),
        NodeGene(NodeTypes.OUTPUT,5)
    ]

    genome_connections = [
        ConnectionGene(0, 0,0,3, 0.5, 1),
        ConnectionGene(1, 0,1,3, 0.5, 1),
        ConnectionGene(2, 0,1,4, 0.5, 1),
        ConnectionGene(3, 0,2,4, 0.5, 1),
        ConnectionGene(4, 0,3,4, 0.5, 1),
        ConnectionGene(5, 0,4,5, 0.5, 1),
        ConnectionGene(6, 0,1,5, 0.5, 1)
    ]

    network = compiler(genome_nodes,genome_connections)

    network.print()

    x = jnp.array([1.,1.,1.]) # output should be 1.25 for three times 1. for this network
    out = network.activate(x)
    print(f"result: {out}")
    assert out == 1.25


def test_compilation_of_network_2():
    genome_nodes = [
        NodeGene(NodeTypes.INPUT,0),
        NodeGene(NodeTypes.INPUT,1),
        NodeGene(NodeTypes.INPUT,2),
        NodeGene(NodeTypes.HIDDEN,3),
        NodeGene(NodeTypes.HIDDEN,4),
        NodeGene(NodeTypes.OUTPUT,5)
    ]

    genome_connections = [
        ConnectionGene(0,0,0,3, 0.5, 1),
        ConnectionGene(1,0,1,3, 0.5, 1),
        ConnectionGene(2,0,1,4, 0.5, 1),
        ConnectionGene(3,0,2,4, 0.5, 1),
        ConnectionGene(4,0,3,5, 0.5, 1),
        ConnectionGene(5,0,4,5, 0.5, 1)
    ]

    network = compiler(genome_nodes,genome_connections)

    x = jnp.array([1.,1.,1.]) # output should be 1.25 for three times 1. for this network
    out = network.activate(x)
    print(f"result: {out}")
    assert out == 1
