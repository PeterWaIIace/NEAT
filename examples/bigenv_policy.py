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

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BigEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(BigEnv, self).__init__()
        
        # Observation space: shape (4, 64)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4, 64), dtype=np.float32
        )
        
        # Action space: discrete 4x4 grid → represented as MultiDiscrete
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4, 4), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        # Action is something like [2, 1]
        # Dummy transition: next state = random
        self.state = self.observation_space.sample()
        
        reward = np.random.rand()  # dummy reward
        terminated = False
        truncated = False
        info = {}
        
        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Rendering state with shape {self.state.shape}")

    def close(self):
        pass


def main():

    gym.register(
        id="BigEnv-v0",
        entry_point="__main__:BigEnv",  # same file
    )

    model = PNEAT()

    env = gym.make('BigEnv-v0',render_mode='human')

    model.set_env(env)
    model.learn(2_000_000)

if __name__=="__main__":
    main()