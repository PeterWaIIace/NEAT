import gymnasium as gym
import numpy as np
import argparse
import pickle 
import time
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')
from src.neat import NEAT

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--input_file', '-i', type=str, default='', help='Input file path')
parser.add_argument('--epoch', '-e', type=int, default=0, help='Starting epoch number')

N = 20
MATCHES = 3
GENERATIONS = 100
POPULATION_SIZE = 5
NMC = 0.9
CMC = 0.9
WMC = 0.9
BMC = 0.9
AMC = 0.9
δ_th = 5
MUTATE_RATE = 1
RENDER_HUMAN = True
epsylon = 0.2

INPUT_SIZE = 3
OUTPUT_SIZE = 2
keep_top = 2

def mutate(neat):
    # EVOLVE EVERYTHING
    neat.cross_over(δ_th = δ_th, N = N)
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

    # env = gym.make("GymV21Environment-v0", env=env, apply_api_compatibility=True)
    if args.input_file:
        my_neat = pickle.load(open(args.input_file,"br"))
    else:
        my_neat = NEAT(INPUT_SIZE,
                       OUTPUT_SIZE,
                       POPULATION_SIZE,
                    keep_top = keep_top,
                    nmc = 0.5,
                    cmc = 0.5,
                    wmc = 0.5,
                    bmc = 0.5,
                    amc = 0.5,
                    N = N,
                    δ_th = δ_th)

    starting_epoch = args.epoch

    models_path = "models"

    env = gym.make('Pendulum-v1',render_mode='human',g=9.81)

    for e in range(starting_epoch,starting_epoch + GENERATIONS, 1):
        all_rewards = []

        my_neat = mutate(my_neat)
        networks = my_neat.evaluate()

        print(f"================ EPOCH: {e} ================")
        total_elapsed_time = 0
        for n,network in enumerate(networks):
            start_time = time.time()
            
            total_reward = 0
            for _ in range(MATCHES):
                observation,_ = env.reset()
                match_reward = 0
                done = False
                for _ in range(50):
                    actions = network.activate(observation)
                    actions = np.array([-actions[0],actions[1]])
                    # take biggest value index and make it action performed
                    observation, reward, trunacted, terminated ,info = env.step(actions)
                    match_reward += reward
                    done = trunacted or terminated
                total_reward += match_reward

            total_reward = total_reward/MATCHES
            all_rewards.append(total_reward)
            total_elapsed_time += time.time() - start_time
            print(f"network: {n} reward: {total_reward} elapsed time: {time.time() - start_time}s")
    
        my_neat.update(all_rewards)
        
        avg_fitness = np.sum(all_rewards)/len(all_rewards)
        print(f"Average fitness: {avg_fitness} total_elapsed_time: {total_elapsed_time/len(all_rewards)}")
        pickle.dump(my_neat,open(f"{models_path}/neat_epoch_{e}_{GENERATIONS}_{POPULATION_SIZE}.model","bw"))


    env.close()

if __name__=="__main__":
    main()