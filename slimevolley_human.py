import gymnasium as gym
import slimevolleygym
import numpy as np
from pynput import keyboard

oldEnv = slimevolleygym.SlimeVolleyEnv()
oldEnv.reset()
env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True, render_mode="human")
obs = env.reset()
done = False
total_reward = 0
action = [0,0,0]

def on_press(key):
    global action
    if key == keyboard.Key.left:
        action = [0.458, 0.01 ,0.   ]  # Example action for moving left
    elif key == keyboard.Key.right:
        action = [0, 0.6, 0.01]  # Example action for moving right
    elif key == keyboard.Key.up:
        action = [0, 0, 0.3]  # Example action for jumping

def on_release(key):
    global action
    if key in [keyboard.Key.left, keyboard.Key.right, keyboard.Key.up]:
        action = [0, 0, 0]  # Reset the action when keys are released

# Set up keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    action_t = [0] * 3
    if np.sum(action) > 0:
        action_t[np.argmax(action)] = 1
    observation, reward, terminated, truncated, info = env.step(action_t)
    total_reward += reward
    done = terminated or truncated

print("score:", total_reward)