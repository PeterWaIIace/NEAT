import gymnasium as gym
import slimevolleygym
from neat import NEAT
import jax.numpy as jnp
import numpy as np
import argparse
import pickle 
import random
import time
import csv
import sys
import os

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--input_file', '-i', type=str, default='', help='Input file path')

class Tournament:

    def __init__(self,players):
        self.players = players
        self.points = [0]*len(players)
        self.pairs = []
        self.played = [] 
        self.__make_tournament()
        pass

    def generate_pairs(self):
        raise NotImplementedError

    def __make_tournament(self):
        self.generate_pairs()

    def get_next_pair(self):
        for pair in self.pairs:
            yield pair  

    def add_points(self,player,points):
        i = self.players.index(player)
        self.points[i] += points


    def get_points(self,player):
        i = self.players.index(player)
        return self.points[i]
    
    def get_score_table(self):
        return self.points
        
    def show_result(self,player):
        index = self.players.index(player)
        print(f"player: {index }, scores: {self.get_points(player)}")

    def get_player_index(self,player):
        return self.players.index(player)
    
    def show_score_table(self):
        print("====== SCORE  TABLE ======")
        for player in self.players:
            self.show_result(player)
        print("====== END OF TABLE ======")

class RoundRobin(Tournament):

    # generate every player play with every other player
    def generate_pairs(self):
        self.pairs = []
        players_cp = self.players.copy()
        random.shuffle(players_cp)
        for n,player_1 in enumerate(players_cp):
            for player_2 in players_cp[n+1:]:
                self.pairs.append((player_1,player_2))

N = 10
EPOCHS = 20
POPULATION_SIZE = 10
NMC = 0.5
CMC = 0.5
WMC = 0.5
BMC = 0.5
AMC = 0.5
δ_th = 5
MUTATE_RATE = 16
RENDER_HUMAN = False


def main():
    my_neat = NEAT(12,3,
                POPULATION_SIZE,
                N = N,
                nmc = NMC,
                cmc = CMC,
                wmc = WMC,
                bmc = BMC,
                amc = AMC,
                δ_th = δ_th)

    args = parser.parse_args()

    oldEnv = slimevolleygym.SlimeVolleyEnv()
    oldEnv.survival_bonus = True
    oldEnv.reset()
    if RENDER_HUMAN :
        env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True,render_mode = "human")
    else:
        env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True)

    input_file = None
    if args.input_file:
        input_file = args.input_file
        my_neat.load_population(input_file)
        input_file = "loaded" #WARNING: reusing variable

    models_path = "models"
    game = f"slimevolleygym_multi_mutate_{MUTATE_RATE}_δ_th{δ_th}_S{POPULATION_SIZE}_N{N}_surv_{input_file}"
    
    for e in range(EPOCHS):
        start_time = time.time()
        print(f"================ EPOCH: {e} ================")
        os.makedirs(f"{models_path}/rest_{game}_{e}", exist_ok=True)    

        my_neat.evolve(MUTATE_RATE)
        networks = my_neat.evaluate()

        tournament = RoundRobin(networks)
        print(f"prepareing phase took: {(time.time() - start_time)}s")
        start_time = time.time()
        for network_1,network_2 in tournament.get_next_pair():

            observation, info = env.reset()
            observation = observation[0]
            observation2 = observation

            done = False

            while not done:
                actions1 = network_1.activate(observation)
                actions1 = np.round(actions1 + 0.5).astype(int)

                actions2 = network_2.activate(observation2)
                actions2 = np.round(actions2 + 0.5).astype(int)

                observation, reward, done, info = oldEnv.step(actions1,otherAction = actions2)
                observation2 = info['otherObs']
                
                tournament.add_points(network_1,reward)
                tournament.add_points(network_2,-reward)

                if RENDER_HUMAN :
                    oldEnv.render()

            pickle.dump(network_1.dump_genomes(),open(f"{models_path}/rest_{game}_{e}/{game}_e{e}_n{tournament.get_player_index(network_1)}.neatpy","wb"))
            pickle.dump(network_2.dump_genomes(),open(f"{models_path}/rest_{game}_{e}/{game}_e{e}_n{tournament.get_player_index(network_2)}.neatpy","wb"))
            network_1.visualize(f"rest_{game}_{e}/{game}_e{e}_n{tournament.get_player_index(network_1)}")
            network_2.visualize(f"rest_{game}_{e}/{game}_e{e}_n{tournament.get_player_index(network_2)}")

        tournament.show_score_table()

        scores = tournament.get_score_table()
        avg_fitness = np.sum(scores)/len(scores)
        print(f"elapsed time: {(time.time() - start_time)}s average game time: {(time.time() - start_time)/len(scores)}s")
        print(f"Average fitness: {avg_fitness}")
        my_neat.update(scores)

        print(f"species after: {np.max(my_neat.species)} and networks: {len(networks)}")
        if np.max(my_neat.species) >= len(networks) - 1:
            print(f"prunning everything below: {avg_fitness}")
            my_neat.prune(avg_fitness)
        
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
        index = scores.index(max(scores))
        network = networks[index]
        pickle.dump(network.dump_genomes(),open(f"{models_path}/{game}_e{e}_best.neatpy","wb"))
        network.visualize(f"{game}_e{e}_best")

    env.close()

import resource

if __name__=="__main__":
    main()