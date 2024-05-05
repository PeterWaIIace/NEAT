import gymnasium as gym
import slimevolleygym
from neat import NEAT
import jax.numpy as jnp
import numpy as np
import argparse
import pickle 
import time
import csv
import sys
import os

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--input_file', '-i', type=str, default='', help='Input file path')
parser.add_argument('--epoch', '-e', type=int, default=0, help='Starting epoch number')

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
    input_file = None
    args = parser.parse_args()

    # env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True)
    if args.input_file:
        my_neat = pickle.load(open(args.input_file,"br"))
    else:
        my_neat = NEAT(12,3,POPULATION_SIZE,
                    nmc = 0.5,
                    cmc = 0.5,
                    wmc = 0.5,
                    bmc = 0.5,
                    amc = 0.5,
                    N = N,
                    δ_th = δ_th)

    starting_epoch = args.epoch

    models_path = "models"

    oldEnv = slimevolleygym.SlimeVolleyEnv()
    oldEnv.survival_bonus = True

    for e in range(starting_epoch,starting_epoch + GENERATIONS, 1):
        all_rewards = []

        my_neat = mutate(my_neat)
        networks = my_neat.evaluate()

        print(f"================ EPOCH: {e} ================")
        total_elapsed_time = 0
        for n,network in enumerate(networks):
            start_time = time.time()
            
            observation = oldEnv.reset()
            done = False
            total_reward = 0
            while not done:
                
                actions = np.array(network.activate(observation))
                # take biggest value index and make it action performed
                action_t = [0] * 3
                actions[actions < 0.0] = 0.0
                if np.sum(actions) != 0:
                    action_t[np.argmax(actions)] = 1
                observation, reward, done, info = oldEnv.step(action_t)
                total_reward += reward
                if RENDER_HUMAN:
                    oldEnv.render()

            all_rewards.append(total_reward)
    
            total_elapsed_time += time.time() - start_time
    
        my_neat.update(all_rewards)
        
        avg_fitness = np.sum(all_rewards)/len(all_rewards)
        print(f"Average fitness: {avg_fitness} total_elapsed_time: {total_elapsed_time/len(all_rewards)}")
        pickle.dump(my_neat,open(f"{models_path}/neat_epoch_{e}_{GENERATIONS}_{POPULATION_SIZE}.model","bw"))

    loaded_neat = pickle.load(open("neat.model","br"))

    oldEnv.close()

if __name__=="__main__":
    main()