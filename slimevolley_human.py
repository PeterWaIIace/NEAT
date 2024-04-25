import gymnasium as gym
import slimevolleygym

oldEnv = slimevolleygym.SlimeVolleyEnv()
oldEnv.reset()
env = gym.make("GymV21Environment-v0", env=oldEnv, apply_api_compatibility=True, render_mode="human")
obs = env.reset()
done = False
total_reward = 0
action = [1,0,1]

while not done:
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print("score:", total_reward)