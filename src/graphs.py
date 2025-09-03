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
from src.nn import Node
# First networkx library is imported
# along with matplotlib

def graph(nodes, n, rows, cols):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    node_color_map = {
        NodeTypes.INPUT.value: "skyblue",
        NodeTypes.NODE.value: "lightgreen",
        NodeTypes.OUTPUT.value: "salmon",
    }
    for node in nodes:
        index = int(node.index)
        G.add_node(index, color=node_color_map[int(node.type)])

    # Add edges to the graph based on inputs
    for node in nodes:
        for input in node.inputs:
            index = int(node.index)
            input = int(input.index)
            G.add_edge(input, index)

    # Draw the graph
    # Create subplot
    ax = plt.subplot(rows, cols, n + 1)  # ← this makes subplot n
    pos = nx.spring_layout(G)  # Layout for better visualization
    node_colors = [G.nodes[i]["color"] for i in G.nodes]

    nx.draw(G, pos, ax=ax, with_labels=True, node_size=10, node_color=node_colors, font_size=2, font_weight="bold")
    ax.set_title(f"Graph {n}")
    plt.pause(0.01)


def graphProcessor(genome):
    """compile your network into FF network"""
    # I need to make sure that all output neurons are at the same layer
    ngenomes, cgenomes = genome.nodes, genome.connections
    nodes = []
    active_nodes = ngenomes[ngenomes[:, Genome.N_EN] != 0.0]
    for _, node in enumerate(active_nodes):
        nodes.append(Node(node[Genome.N_INDEX], node[Genome.N_TYPE], node[Genome.N_BIAS], node[Genome.N_ACT]))

    for c in cgenomes[cgenomes[:, Genome.enabled] != 0.0]:
        nodes[int(c[Genome.C_OUT])].add_input(nodes[int(c[Genome.C_IN])], c[Genome.C_W])

    changes = True
    while changes:
        changes = False
        used_nodes = []
        for node in nodes:
            if len(node.inputs) > 0 or node.type != NodeTypes.NODE.value:
                used_nodes.append(node)
            else:
                changes = True
        nodes = used_nodes

        for node in used_nodes:
            nodes_to_remove = []
            for input in node.inputs:
                if input not in used_nodes:
                    nodes_to_remove.append(input)

            for rm_node in nodes_to_remove:
                node.rm_input(rm_node)
                changes = True

        # for node in used_nodes:
        #     print(f"{[int(input.index) for input in node.inputs],int(node.index)}")
        # print("-----------")
    # nodes = topologicalSort(nodes)
    return nodes
