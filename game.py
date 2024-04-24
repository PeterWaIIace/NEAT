# import gymnasium as gym
import gym
import slimevolleygym
from neat import NEAT

def run():

    # env = gym.make("Acrobot-v1", render_mode="human")
    # env = gym.make("Acrobot-v1")
    my_neat = NEAT(12,3, 20)


    epochs = 50
    prev_action = 0.0
    experiment_length = 100
    models_path = "models"
    game = "SlimeVolley-v0"
    for e in range(epochs):
        print(f"================ EPOCH: {e} ================")
        env = gym.make(game,apply_api_compatibility=True)
        observation, info = env.reset(seed=42)
        all_rewards = []
        my_neat.evolve()

        networks = my_neat.evaluate()
        for n,network in enumerate(networks):
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

            all_rewards.append(total_reward)
            print(f"net: {n}, fitness: {total_reward}")

        env.close()

        #display the best:
        index = all_rewards.index(max(all_rewards))
        network = networks[index]
        env = gym.make(game, render_mode="human")

        print(f"Displaying the best: {index}, max reward: {max(all_rewards)}")
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

        pickle.dump(network,open(f"{models_path}/{game}_e{e}.neatpy","wb"))
        my_neat.update(all_rewards)

        env.close()

if __name__=="__main__":
    run()