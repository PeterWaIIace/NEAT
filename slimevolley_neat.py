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

N = 20
GENERATIONS = 100
POPULATION_SIZE = 20
NMC = 0.5
CMC = 0.5
WMC = 0.5
BMC = 0.5
AMC = 0.5
δ_th = 5
MUTATE_RATE = 16
RENDER_HUMAN = True
epsylon = 0.5

def mutate(neat):
    # EVOLVE EVERYTHING
    neat.mutate_weight(epsylon = epsylon,wmc=WMC)
    neat.mutate_bias(epsylon = epsylon,bmc=BMC)
    for _ in range(MUTATE_RATE):
        neat.mutate_activation(amc=AMC)
        neat.mutate_nodes(nmc=NMC)
        neat.mutate_connections(cmc=CMC)
    neat.cross_over(δ_th = δ_th, N = N)
    return neat
    

def main():
    input_file = None
    args = parser.parse_args()

    oldEnv = slimevolleygym.SlimeVolleyEnv()
    oldEnv.survival_bonus = True
    env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True, render_mode="human")

    total_reward = 0

    my_neat = NEAT(12,3,POPULATION_SIZE,
                nmc = 0.5,
                cmc = 0.5,
                wmc = 0.5,
                bmc = 0.5,
                amc = 0.5,
                N = N,
                δ_th = δ_th)

    if args.input_file:
        input_file = args.input_file
        my_neat.load_population(input_file)
        input_file = "loaded" #WARNING: reusing variable

    models_path = "models"
    game = f"slimevolleygym_mutate_{MUTATE_RATE}_δ_th{δ_th}_S{POPULATION_SIZE}_N{N}_surv_{input_file}"
    
    for e in range(GENERATIONS):
        print(f"================ EPOCH: {e} ================")
        all_rewards = []
        os.makedirs(f"{models_path}/rest_{game}_{e}", exist_ok=True)    

        my_neat = mutate(my_neat)
        networks = my_neat.evaluate()

        for n,network in enumerate(networks):
            
            observation, info = env.reset()
            observation = observation[0]
            done = False
            total_reward = 0
            while not done:
                
                actions = np.array(network.activate(observation))
                # take biggest value index and make it action performed
                action_t = [0] * 3
                actions[actions < 0.0] = 0.0
                if np.sum(actions) != 0:
                    action_t[np.argmax(actions)] = 1

                print(observation)
                print(actions,action_t)
                observation, reward, done, _, info = env.step(action_t)
                total_reward += reward

            all_rewards.append(total_reward)
            print(f"net: {n}, fitness: {total_reward}")

            pickle.dump(network.dump_genomes(),open(f"{models_path}/rest_{game}_{e}/{game}_e{e}_n{n}.neatpy","wb"))
            network.visualize(f"rest_{game}_{e}/{game}_e{e}_n{n}")

        avg_fitness = np.sum(all_rewards)/len(all_rewards)
        print(f"Average fitness: {np.sum(all_rewards)/len(all_rewards)}")
        my_neat.update(all_rewards)
        
        params = my_neat.get_params()
        # List of keys to define the order of columns in the CSV
        keys = ["net","specienumber", "fitness", "connections", "nodes", "nmc", "cmc", "wmc", "bmc", "amc", "C1", "C2", "C3", "N"]
        # Write data to CSV file
        with open(f"{models_path}/csv_params_{game}_e{e}", mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)            
            # Write header
            writer.writeheader()
            for p in params:
                # Write data
                writer.writerow(p)

        # save the best
        index = all_rewards.index(max(all_rewards))
        network = networks[index]
        pickle.dump(network.dump_genomes(),open(f"{models_path}/{game}_e{e}_best.neatpy","wb"))
        network.visualize(f"{game}_e{e}_best")

    env.close()

if __name__=="__main__":
    main()