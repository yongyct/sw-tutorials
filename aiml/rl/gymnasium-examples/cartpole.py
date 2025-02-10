import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import time

# Initialize the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Initialize Pygame for rendering
pygame.init()
clock = pygame.time.Clock()

# Set display dimensions
display_width, display_height = 600, 400
screen = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("CartPole Visualization")

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
min_epsilon = 0.01
epsilon_decay = 0.995
num_episodes = 1000

def discretize_state(state, bins):
    """Discretize continuous state space into bins."""
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_bounds[1] = [-3.0, 3.0]  # Clip cart velocity
    state_bounds[3] = [-math.radians(50), math.radians(50)]  # Clip pole angular velocity

    state_indices = []
    for i, value in enumerate(state):
        # Handle out-of-bound state values
        if value < state_bounds[i][0]:
            index = 0
        elif value > state_bounds[i][1]:
            index = bins - 1
        else:
            # Map continuous state to discrete bucket index
            scale = (value - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])
            index = int(scale * (bins - 1))
        state_indices.append(index)
    return tuple(state_indices)

# Define action space and state bins
n_bins = 24
q_table = np.zeros((n_bins,) * env.observation_space.shape[0] + (env.action_space.n,))

rewards_list = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state, n_bins)
    total_reward = 0

    for step in range(200):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take the action and observe the next state and reward
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = discretize_state(next_state, n_bins)

        # Update Q-value using Bellman equation
        max_next_q = np.max(q_table[next_state])
        q_table[state + (action,)] += alpha * (reward + gamma * max_next_q - q_table[state + (action,)])

        state = next_state
        total_reward += reward

        # Pygame visualization
        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        frame_surface = pygame.transform.scale(frame_surface, (display_width, display_height))
        screen.blit(frame_surface, (0, 0))
        pygame.display.update()
        clock.tick(30)  # Limit to 30 FPS

        if done or truncated:
            break

    # Decay epsilon for less exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    rewards_list.append(total_reward)

    # Log training progress
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Average Reward (last 100 episodes): {np.mean(rewards_list[-100:]):.2f}")

# Plot the learning curve
plt.plot(rewards_list)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning on CartPole")
plt.show()

# Clean up
env.close()
pygame.quit()

