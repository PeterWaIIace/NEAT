
import jax.numpy as jnp
import numpy as np
import random

from enum import Enum
from utils.random import StatefulRandomGenerator

Rnd = StatefulRandomGenerator()

class NodeTypes(Enum):
    NODE = 1
    INPUT = 2
    OUTPUT = 3

class Genome:

    CHUNK = 20

    CONNECTION_SIZE = 5
    C_INNOV = 0
    C_IN = 1
    C_OUT = 2
    C_W = 3
    enabled = 4

    NODE_SIZE = 5
    N_INDEX = 0
    N_TYPE = 1
    N_BIAS = 2
    N_ACT = 3
    N_EN = 4
    
    MAX_NODES = 10000

    def __init__(self):
        self.index = 0
        self.specie = 0
        self.fitness = 1.0
        self.max_innov = 0

        self.current_size = 0
        self.conn_lookup = set()
        self.conn_exists = jnp.zeros((Genome.MAX_NODES, Genome.MAX_NODES), dtype=bool)

        self.connections = jnp.zeros(
            (self.CHUNK, self.CONNECTION_SIZE),
        )
        self.nodes = jnp.zeros(
            (self.CHUNK, self.NODE_SIZE),
        )

    def check_against(self, genome):
        """check agaisnt other genome and make them length equal if longer"""
        if len(genome.connections) > len(self.connections):
            diff = len(genome.connections) - len(self.connections)
            self.connections = jnp.concatenate(
                (
                    self.connections,
                    jnp.zeros(
                        (diff, self.CONNECTION_SIZE),
                    ),
                ),
                axis=0,
            )

        if len(genome.nodes) > len(self.nodes):
            diff = len(genome.nodes) - len(self.nodes)
            self.nodes = jnp.concatenate(
                (
                    self.nodes,
                    jnp.zeros(
                        (diff, self.NODE_SIZE),
                    ),
                ),
                axis=0,
            )

    def __get_possible_conn(self):
        # Active nodes
        active_mask = self.nodes[:, self.N_EN] != 0
        active_nodes = self.nodes[active_mask]

        # Split by type
        input_nodes = active_nodes[active_nodes[:, self.N_TYPE] != float(NodeTypes.OUTPUT.value), self.N_INDEX].astype(int)
        output_nodes = active_nodes[active_nodes[:, self.N_TYPE] != float(NodeTypes.INPUT.value), self.N_INDEX].astype(int)

        # Cartesian product (broadcasted, no big meshgrid)
        in_expanded = jnp.repeat(input_nodes, len(output_nodes))
        out_expanded = jnp.tile(output_nodes, len(input_nodes))
        pairs = np.stack([in_expanded, out_expanded], axis=1)

        # Remove self-connections
        mask = pairs[:, 0] != pairs[:, 1]

        # Remove existing connections (use adjacency matrix instead of Python loop)
        mask &= ~self.conn_exists[pairs[:, 0], pairs[:, 1]]

        pairs = pairs[mask]

        return pairs

    def add_node(self, type: int, bias: float, act: int):
        """Adding node"""

        # Keep a separate counter instead of recomputing
        new_node_index = self.current_size
        self.current_size += 1

        # Resize only if needed (rare, amortized cost)
        if new_node_index >= self.nodes.shape[0]:
            self.nodes = jnp.concatenate(
                (self.nodes, jnp.zeros((self.CHUNK, self.NODE_SIZE))),
                axis=0,
            )

        new_node_values = jnp.array([new_node_index, type, bias, act, 1.0])
        self.nodes = self.nodes.at[new_node_index].set(new_node_values)

        return new_node_index

    def add_connection(self, innov: int, in_node: int, out_node: int, weight: float):
        """Adding connection"""

        # Quick skip if invalid or already exists
        if (
            self.nodes[in_node, self.N_EN] == 0
            or self.nodes[out_node, self.N_EN] == 0
            or (in_node, out_node) in self.conn_lookup  # Python-side set for fast lookup
        ):
            return innov

        # Resize only if needed
        if innov >= self.connections.shape[0]:
            self.connections = jnp.concatenate(
                (self.connections, jnp.zeros((self.CHUNK, self.CONNECTION_SIZE))),
                axis=0,
            )

        # Insert connection
        new_connection_values = jnp.array([innov, in_node, out_node, weight, 1.0])
        self.connections = self.connections.at[innov].set(new_connection_values)

        # Update lookup and innov counter
        self.conn_lookup.add((in_node, out_node))
        self.conn_exists = self.conn_exists.at[in_node, out_node].set(True)
        self.max_innov = innov + 1

        return self.max_innov

    def add_r_connection(self, innov):

        pairs = self.__get_possible_conn()
        if len(pairs) <= 0:
            return innov

        random.shuffle(pairs)

        pair = pairs[0]
        in_node = pair[0]
        out_node = pair[1]

        innov = self.add_connection(int(innov), int(in_node), int(out_node), 1.0)
        return innov

    def add_r_node(self, innov: int):
        exisitng_connections = self.connections[self.connections[:, self.enabled] != 0]

        chosen_connection = exisitng_connections[Rnd.randint(max=len(exisitng_connections))]
        self.connections = self.connections.at[int(chosen_connection[self.C_INNOV]), self.enabled].set(0.0)

        new_node = self.add_node(NodeTypes.NODE.value, 0.0, 0)
        in_node = int(chosen_connection[self.C_IN])

        innov = self.add_connection(innov, in_node, int(new_node), chosen_connection[self.C_W])

        out_node = int(chosen_connection[self.C_OUT])
        innov = self.add_connection(innov, int(new_node), out_node, chosen_connection[self.C_W])
        return innov

    def change_weigth(self, weigth):
        self.connections = self.connections.at[:, self.C_W].add(weigth)

    def change_bias(self, bias):
        self.nodes = self.nodes.at[:, self.N_BIAS].add(bias)

    def change_activation(self, act):
        self.nodes = self.nodes.at[:, self.N_ACT].set(act)

