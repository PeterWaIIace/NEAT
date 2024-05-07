import gymnasium as gym
import slimevolleygym
from neat import NEAT
import jax.numpy as jnp
import numpy as np
import argparse
import pickle 
import time

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--input_file', '-i', type=str, default='', help='Input file path')
parser.add_argument('--epoch', '-e', type=int, default=0, help='Starting epoch number')

N = 20
MATCHES = 3
GENERATIONS = 100
POPULATION_SIZE = 20
NMC = 0.8
CMC = 0.8
WMC = 0.8
BMC = 0.8
AMC = 0.8
δ_th = 5
MUTATE_RATE = 1
RENDER_HUMAN = True
epsylon = 0.2

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

    # env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True)
    if args.input_file:
        my_neat = pickle.load(open(args.input_file,"br"))
    else:
        my_neat = NEAT(12,3,POPULATION_SIZE,
                    keep_top = 4,
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
            
            total_reward = 0
            for _ in range(MATCHES):
                observation = oldEnv.reset()
                done = False       
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

            all_rewards.append(total_reward/MATCHES)
            print(f"network: {n} reward: {total_reward/MATCHES} elapsed time: {time.time() - start_time}s")
    
            total_elapsed_time += time.time() - start_time
    
        my_neat.update(all_rewards)
        
        avg_fitness = np.sum(all_rewards)/len(all_rewards)
        green = '\033[92m'
        reset = '\033[0m'
        print(f"{green}Average fitness: {avg_fitness} total_elapsed_time: {total_elapsed_time/len(all_rewards)}s{reset}")
        pickle.dump(my_neat,open(f"{models_path}/neat_epoch_{e}_{GENERATIONS}_{POPULATION_SIZE}.model","bw"))


    oldEnv.close()

if __name__=="__main__":
    main()