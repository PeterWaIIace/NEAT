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

def main():
    input_file = None
    args = parser.parse_args()


    oldEnv = slimevolleygym.SlimeVolleyEnv()
    oldEnv.survival_bonus = True
    oldEnv.reset()
    env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True, render_mode="human")
    obs = env.reset()
    total_reward = 0

    δ_th = 5
    N = 20
    POPULATION_SIZE = 10
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

    epochs = 20
    models_path = "models"
    evo_rate = 4
    game = f"slimevolleygym_mutate_{evo_rate}_δ_th{δ_th}_S{POPULATION_SIZE}_N{N}_surv_{input_file}"
    
    for e in range(epochs):
        print(f"================ EPOCH: {e} ================")
        all_rewards = []
        my_neat.evolve(evo_rate)

        os.makedirs(f"{models_path}/rest_{game}_{e}", exist_ok=True)    
        networks = my_neat.evaluate()

        for n,network in enumerate(networks):
            
            observation, info = env.reset()
            observation = observation[0]
            done = False
            total_reward = 0
            while not done:
                actions = network.activate(observation)
                actions = np.round(actions + 0.5).astype(int)
                observation, reward, done, _, info = env.step(actions)
                total_reward += reward

            all_rewards.append(total_reward)
            print(f"net: {n}, fitness: {total_reward}")

            pickle.dump(network.dump_genomes(),open(f"{models_path}/rest_{game}_{e}/{game}_e{e}_n{n}.neatpy","wb"))
            network.visualize(f"rest_{game}_{e}/{game}_e{e}_n{n}")

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