import gymnasium as gym
import pygame
import numpy as np

def get_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return 2  # Apply force to move the joint clockwise
    elif keys[pygame.K_DOWN]:
        return 1  # Apply force to move the joint anti-clockwise
    else:
        return 0  # Apply no force

if __name__ == "__main__":
    env = gym.make("Acrobot-v1", render_mode="human")

    pygame.init()
    screen = pygame.display.set_mode((600, 400))

    observation = env.reset()

    running = True
    total_reward = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = get_action()
        ret = env.step(action)
        print(ret,len(ret))
        observation, reward, terminated, info,_ = ret
        total_reward += reward
        if terminated:
            print(f"Episode terminated. {total_reward}")
            break

    env.close()
    pygame.quit()
