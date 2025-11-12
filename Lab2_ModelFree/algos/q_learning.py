import os
import random
from collections import defaultdict
from env.gridworld_env import Action
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from algos.utils import plot_value_and_policy

def q_learning(env, save_name, episodes=1000, alpha=0.1, gamma=0.99, epsilon=1.0, render=False, log_interval=100):
    Q = defaultdict(lambda: {a: 0.0 for a in Action})
    
    reward_history = []
    success_rate_history = []
    success_count = 0
    max_steps = 500

    # epsilon decay settings
    initial_epsilon = epsilon
    min_epsilon = 0.05
    decay_rate = 0.99

    for episode in tqdm(range(episodes), desc="Training Q-Learning"):
        # epsilon decay (initially, high epsilon to encourage exploration)
        epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** episode))

        state = tuple(env.reset())
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            if render and episode % 500 == 0:
                env.render()
                time.sleep(0.05)

            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(list(Action))
            else:
                action = max(Q[state], key=Q[state].get)

            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)

            # Reward update: explicitly reinforce when goal is reached
            if done and reward == 100:
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                max_next = max(Q[next_state].values())
                Q[state][action] += alpha * (reward + gamma * max_next - Q[state][action])

            state = next_state
            episode_reward += reward
            if reward == 100:
                success_count += 1

            steps += 1

        reward_history.append(episode_reward)
        output_folder = f"./outputs/q_learning_{save_name}"
        if (episode + 1) % log_interval == 0 or episode == episodes - 1:
            V = {s: max(Q[s].values()) for s in Q}
            policy = {s: max(Q[s], key=Q[s].get) for s in Q}
            plot_value_and_policy(V, policy, env.grid, episode, env.width, env.height, output_folder)

            avg_reward = sum(reward_history[-log_interval:]) / log_interval
            success_rate = success_count
            success_rate_history.append(success_rate)
            print(f"[Ep {episode+1}] Avg Reward: {avg_reward:.2f}, Successes: {success_rate}")
            success_count = 0

    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}

    #plot rewards
    smoothing = True
    smoothed_rewards = []
    window_size = 10
    if smoothing:
        for i in range(len(reward_history)):
            start_idx = max(0, i - window_size + 1)
            smoothed_rewards.append(sum(reward_history[start_idx:i+1]) / (i - start_idx + 1))
        reward_history = smoothed_rewards
    plt.figure()
    plt.plot(smoothed_rewards if smoothing else reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning - Episode Rewards, init_epsilon = {:.2f}'.format(initial_epsilon))
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'q_learning_rewards.png'))
    plt.close()

    return Q, policy