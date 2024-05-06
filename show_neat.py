from neat import NEAT, Painter, Genome, Neuron
import jax.numpy as jnp
import numpy as np
import argparse
import pickle 

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--input_file', '-i', type=str, default='', help='Input file path')

N = 20
GENERATIONS = 100
POPULATION_SIZE = 20
NMC = 0.5
CMC = 0.5
WMC = 0.5
BMC = 0.5
AMC = 0.5
δ_th = 5
MUTATE_RATE = 4
RENDER_HUMAN = True
epsylon = 0.2

def compile_gen2graph(genome):
    ''' compile your network into FF network '''
    # I need to make sure that all output neurons are at the same layer
    ngenomes, cgenomes = genome.node_gen, genome.con_gen
    neurons = []
    active_nodes = ngenomes[ngenomes[:,0] != 0.0]
    for _,node in enumerate(active_nodes):
        neurons.append(
            Neuron(node)
        )

    for c in cgenomes[cgenomes[:,Genome.enabled] != 0.0]:
        if int(c[Genome.o])-1 < len(neurons) and int(c[Genome.i])-1 < len(neurons):
            neurons[int(c[Genome.o])-1].add_input(
                neurons[int(c[Genome.i])-1],
                c[Genome.w])
            
    return neurons

def main():
    args = parser.parse_args()

    # env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True)
    if args.input_file:
        my_neat = pickle.load(open(args.input_file,"br"))
        painter = Painter()
        for n,network in enumerate(my_neat.evaluate()):
            neurons = compile_gen2graph(network.genome)
            print([(neuron.input_list,neuron.index) for neuron in neurons])
            painter.visualize(network,f"graphs/network_{n}")

if __name__=="__main__":
    main()