import gymnasium as gym
import numpy as np
import argparse
import pickle 
import time
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')

from src.neat import PNEAT

def main():

    model = PNEAT()

    env = gym.make('CartPole-v1',render_mode='human')

    model.set_env(env)
    model.learn(2_000_000)

if __name__=="__main__":
    main()