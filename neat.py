''' Fast neat '''

import jax.numpy as jnp 
import jax.random as jrnd
from enum import Enum

class NodeTypes(Enum):
    NODE = 1 
    INPUT = 2
    OUTPUT = 3 

class StatefulRandomGenerator:

    def __init__(self):
        self.key = jrnd.PRNGKey(0)

    def randint_permutations(self,val_range):
        ''' generate subarray of permutation array from defined values range '''
        rnd_value = jrnd.uniform(self.key)
        rnd_value_int = int(rnd_value * val_range)
        indecies = jrnd.permutation(self.key, val_range)[:rnd_value_int]
        self.key = jrnd.split(self.key,1)[0]
        return indecies

Rnd = StatefulRandomGenerator()

class Genome:

    def __init__(self):
        self.connections_length = 20
        self.nodes_length = 20
        self.max_innov = 0
        # helper indicies names
        self.w = 4

        # Connections genomes is array, rows are genomes, but cols are parameters of that genomes
        self.con_gen  = jnp.zeros((self.connections_length,6),)
        self.node_gen = jnp.zeros((self.nodes_length,2),)

    def add_node(self,index,type):
        ''' Adding node '''

        if self.nodes_length <= index:
            self.nodes_length += 20
            new_nodes_spaces = jnp.zeros((20,6),)
            self.node_gen = jnp.concatenate((self.node_gen,new_nodes_spaces), axis=1)

        self.node_gen.at[index].set(jnp.array([index,type.value]))

    def add_connection(self,innov,in_node,out_node,weight):
        ''' Adding connection '''

        # update innovation if is bigger than current innov of genome
        if self.max_innov < innov:
            self.max_innov = innov

        if self.connections_length <= innov:
            self.nodes_length += 20
            new_connections_spaces = jnp.zeros((20,6),)
            self.con_gen = jnp.concatenate((self.con_gen,new_connections_spaces), axis=1)

        self.con_gen.at[innov].set(jnp.array([innov,innov,in_node,out_node,weight,1.0]))

def δ( genome_1, genome_2, c1=1.0, c2=1.0, c3=1.0, N=1.0):
    ''' calculate compatibility between genomes '''
    D = 0  # disjoint genes
    E = 0  # excess genes

    # check smaller innovation number:
    innovation_thresh = genome_1 if genome_1.max_innov < genome_2.max_innov else genome_2

    D_tmp = jnp.subtract(
        (genome_1.con_gen[:innovation_thresh,0]
        ,genome_2.con_gen[:innovation_thresh,0])
    ) # getting all disjoint connections - if array do not have connection then it will have 0 as innovtaion number
    D = len(D_tmp[D_tmp[:,0] != 0])

    E_tmp = jnp.subtract(
        genome_1.con_gen[innovation_thresh:,0]
        ,genome_2.con_gen[innovation_thresh:,0]
    ) # getting all disjoint connections - if array do not have connection then it will have 0 as innovtaion number
    E = len(E_tmp[E_tmp[:,0] != 0])

    W_1 = genome_1.con_gen[D_tmp[:,0] == 0][genome_1.w]
    W_2 = genome_2.con_gen[D_tmp[:,0] == 0][genome_2.w]
    W_avg = jnp.subtract(W_1,W_2)/W_1.size

    d = (c1 * E) / N + (c2 * D) / N + c3 * W_avg
    return d

def sh(δ,δ_t = 0.2):
    ''' sharing fitness threshold function '''
    return δ < δ_t


def mate(superior : Genome, inferior : Genome):
    ''' mate superior Genome with inferior Genome '''
    # check smaller innovation number:
    innovation_thresh = superior.max_innov if superior.max_innov < inferior.max_innov else inferior.max_innov

    indecies = Rnd.randint_permutations(innovation_thresh)
    inferior.con_gen.at[indecies].set(superior.con_gen[indecies])

    indecies = Rnd.randint_permutations(len(inferior.con_gen[innovation_thresh:])) + innovation_thresh
    inferior.con_gen.at[indecies].set(superior.con_gen[indecies])
    # Lazy but working, copy all nodes not existing in inferior but exisitng in superior
    inferior.node_gen.at[inferior.node_gen[:,0] == 0].set(superior.node_gen[inferior.node_gen[:,0] == 0])

    return inferior

if __name__=="__main__":
    superior = Genome()
    inferior = Genome()

    superior.add_node(0,NodeTypes.INPUT)
    superior.add_node(1,NodeTypes.INPUT)
    superior.add_node(2,NodeTypes.NODE)
    superior.add_node(3,NodeTypes.NODE)
    superior.add_node(4,NodeTypes.NODE)
    superior.add_node(5,NodeTypes.OUTPUT)

    inferior.add_node(0,NodeTypes.INPUT)
    inferior.add_node(1,NodeTypes.INPUT)
    inferior.add_node(2,NodeTypes.NODE)
    inferior.add_node(3,NodeTypes.NODE)
    inferior.add_node(4,NodeTypes.NODE)
    inferior.add_node(5,NodeTypes.OUTPUT)

    superior.add_connection(0,0,2,1.0)
    superior.add_connection(1,1,3,1.0)
    superior.add_connection(2,0,4,1.0)
    superior.add_connection(3,1,4,1.0)
    superior.add_connection(4,2,5,1.0)
    superior.add_connection(5,3,5,1.0)
    superior.add_connection(5,4,5,1.0)

    inferior.add_connection(0,0,2,1.0)
    inferior.add_connection(3,1,4,1.0)
    inferior.add_connection(4,2,5,1.0)
    inferior.add_connection(5,4,5,1.0)

    inferior = mate(superior,inferior)