import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import json
from models.linear_policy import LinearPolicy
def select_action(policy_net, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = policy_net(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def train(env_name='LunarLander-v2', episodes=2000, gamma=0.99, lr=1e-3, max_steps=1000):
    env = gym.make(env_name)
    policy_net = LinearPolicy(input_dim=8, hidden_dim=128, output_dim=4)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)

    total_rewards = []

    try:
        for episode in range(episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]

            log_probs = []
            rewards = []

            done = False
            steps = 0

            while not done and steps < max_steps:
                action, log_prob = select_action(policy_net, state)
                next_state, reward, done, *extra = env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                steps += 1

            total_reward = sum(rewards)
            total_rewards.append(total_reward)

            # Compute normalized returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            loss = sum([-log_prob * R for log_prob, R in zip(log_probs, returns)])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving rewards...")

    finally:
        env.close()
        os.makedirs("lunarlander", exist_ok=True)
        with open("lunarlander/rewards_log_linear.json", "w") as f:
            json.dump(total_rewards, f)
        print("Training complete. Rewards saved to lunarlander/rewards_log_linear.json")

if __name__ == "__main__":
    train()
