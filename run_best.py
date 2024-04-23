import sys
import neat
import pickle 
import jax.numpy as jnp
import gymnasium as gym
from neat import FeedForward, Layer, Neuron

def run():

    env = gym.make("Acrobot-v1", render_mode="human")
    
    models_path = "models"
    network = pickle.load(open(f"{models_path}/{sys.argv[1]}","rb"))
    
    prev_action = 0.0
    experiment_length = 100
    

    observation, info = env.reset()
    total_reward = 0

    for _ in range(experiment_length):
        actions = network.activate(jnp.array(observation))
        action = actions.argmax()
        #promote mobility
        if prev_action != action:
            total_reward += abs(observation[4])/10000 + abs(observation[5])/10000
            prev_action = action

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()

if __name__=="__main__":
    run()
