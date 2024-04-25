import gymnasium as gym
import slimevolleygym
from neat import NEAT
import jax.numpy as jnp
import numpy as np
import pickle 
import time
import csv
import os

oldEnv = slimevolleygym.SlimeVolleyEnv()
oldEnv.reset()
env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True, render_mode="human")
obs = env.reset()
total_reward = 0

my_neat = NEAT(12,3,20,nmc = 0.5, cmc = 0.5, wmc = 0.5, bmc = 0.5, amc = 0.5)


epochs = 50
prev_action = 0.0
experiment_length = 100
models_path = "models"
game = "slimevolleygym"
for e in range(epochs):
    print(f"================ EPOCH: {e} ================")
    all_rewards = []
    my_neat.evolve()

    networks = my_neat.evaluate()

    for n,network in enumerate(networks):
        observation, info = env.reset()
        observation = observation[0]
        done = False
        total_reward = 0
        while not done:
            actions = network.activate(observation)
            actions = np.round(actions + 0.5).astype(int)
            observation, reward, terminated, _, info = env.step(actions)
            total_reward += reward
            done = terminated

        all_rewards.append(total_reward)
        print(f"net: {n}, fitness: {total_reward}")

        os.makedirs(f"{models_path}/rest_{game}_{e}", exist_ok=True)
        pickle.dump(network.dump_genomes(),open(f"{models_path}/rest_{game}_{e}/{game}_e{e}_n{n}.neatpy","wb"))
        network.visualize(f"rest_{game}_{e}/{game}_e{e}_n{n}")

        params = my_neat.get_params()
        # List of keys to define the order of columns in the CSV
        keys = ["net","specienumber", "fitness", "connections", "nodes", "nmc", "cmc", "wmc", "bmc", "amc", "C1", "C2", "C3", "N"]
        # Write data to CSV file
        with open(f"{models_path}/rest_{game}_{e}/csv_params_{game}_e{e}_n{n}", mode='w', newline='') as file:
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
