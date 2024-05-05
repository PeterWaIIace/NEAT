from neat import NEAT, Painter
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

def mutate(neat):
    # EVOLVE EVERYTHING
    neat.cross_over(keep_top = 4,δ_th = δ_th, N = N)
    neat.mutate_weight(epsylon = epsylon,wmc=WMC)
    neat.mutate_bias(epsylon = epsylon,bmc=BMC)
    for _ in range(MUTATE_RATE):
        neat.mutate_activation(amc=AMC)
        neat.mutate_nodes(nmc=NMC)
        neat.mutate_connections(cmc=CMC)
    return neat
    

def main():
    args = parser.parse_args()

    # env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True)
    if args.input_file:
        my_neat = pickle.load(open(args.input_file,"br"))
        painter = Painter()
        for n,network in enumerate(my_neat.evaluate()):
            painter.visualize(network,f"graphs/network_{n}")

if __name__=="__main__":
    main()