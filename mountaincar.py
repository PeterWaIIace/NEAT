import gymnasium as gym
from neat import NEAT
import numpy as np
import argparse
import pickle 
import random
import time

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--input_file', '-i', type=str, default='', help='Input file path')
parser.add_argument('--epoch', '-e', type=int, default=0, help='Starting epoch number')

N = 20
MATCHES = 1
GENERATIONS = 100
POPULATION_SIZE = 10
NMC = 0.5
CMC = 0.5
WMC = 0.9
BMC = 0.9
AMC = 0.9
δ_th = 5
MUTATE_RATE = 1
RENDER_HUMAN = True
epsylon = 0.2

INPUT_SIZE = 3
OUTPUT_SIZE = 3
keep_top = 4

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

    env = gym.make('MountainCar-v0',render_mode='human')

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
                while not done:
                    observation =np.append(observation,[match_reward])
                    actions = np.array(network.activate(observation))
                    # take biggest value index and make it action performed
                    observation, reward, trunacted, terminated ,info = env.step(np.argmax(actions))
                    # reward for getting closer to flag
                    flag_pos = 0.5
                    match_reward += reward + (1/(2*flag_pos - observation[0]))/10 + abs(observation[1])
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