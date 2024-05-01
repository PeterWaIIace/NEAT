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


# Set the print options to display the full array
jnp.set_printoptions(threshold=10000)

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


class AgainstItself(Tournament):

    # generate every player play with every other player
    def generate_pairs(self):
        self.pairs = []
        players_cp = self.players.copy()
        for n,player_1 in enumerate(players_cp):
            self.pairs.append((player_1,player_1))

N = 10
GENERATIONS = 20
POPULATION_SIZE = 1
NMC = 0.5
CMC = 0.5
WMC = 0.5
BMC = 0.5
AMC = 0.5
δ_th = 5
MUTATE_RATE = 16
RENDER_HUMAN = True


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

    # EVOLVE EVERYTHING
    my_neat.mutate_activation(amc=AMC)
    my_neat.mutate_weight(epsylon = 0.1,wmc=WMC)
    my_neat.mutate_bias(epsylon = 0.1,bmc=BMC)
    for _ in range(MUTATE_RATE):
        my_neat.mutate_nodes(nmc=NMC)
        my_neat.mutate_connections(cmc=CMC)
    
    my_neat.cross_over(δ_th = δ_th, N = N)

    network = my_neat.evaluate()[0]
    
    observation, info = env.reset()
    observation = observation[0]

    for n in range(20):
        actions1 = network.activate(observation)
        print(f"actions1: {actions1}")
        
        observation, reward, done, info = oldEnv.step(actions1)
        
        oldEnv.render()
    time.sleep(30)
    env.close()

import resource

if __name__=="__main__":
    main()