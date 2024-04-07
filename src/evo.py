from dataclasses import dataclass
from utils import NodeTypes

class NodeGenome:

    def __init__(self,type,index):
        self.type  = type 
        self.index = index 

class ConnectionGenome:

    def __init__(self,innov,in_index,out_index,weight,enabled):
        self.innov = innov 
        self.in_neuron = in_index 
        self.out_neuron = out_index
        self.weight = weight
        self.enabled = enabled 

@dataclass
class Genome:

    ngenome : [NodeGenome] = []
    cgenome : [ConnectionGenome] = []


    def add_node(self):
        new_node = NodeGenome(NodeTypes.HIDDEN,len(self.ngenome))
        self.ngenome.append(new_node)
        add_connection(self,in_node,out_node,weight)
        add_connection(self,in_node,out_node,weight)
        
    def add_connection(self,in_node,out_node,weight):
        self.cgenome.append(ConnectionGenome(0,in_node,out_node,weight,1))
        

class Evomixer:

    def __init__(self,population_size = 50):

        self.genomes = [] 
        self.population_size = population_size
        pass

