#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.distributions import Dirichlet
import matplotlib.pyplot as plt

# Create environment - modified to handle tensors directly
class PortfolioEnv:
    def __init__(self, data):
        self.data = data.detach().clone() if isinstance(data, torch.Tensor) else torch.FloatTensor(data)
        self.n_steps, self.n_assets = self.data.shape
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step].detach().numpy()  # Return numpy array

    def step(self, action):
        return_ = np.dot(action, self.data[self.current_step].numpy())  # Use numpy values
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        next_state = self.data[self.current_step].numpy() if not done else None
        return return_, next_state, done

# Modified RLAgent to handle tensor states
class RLAgent:
    def __init__(self, n_assets):
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(n_assets, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_assets)
        )
        for layer in self.policy:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0.1)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001, weight_decay=1e-4)
        self.gamma = 0.99

    def get_action(self, state):
        state_tensor = state if isinstance(state, torch.Tensor) else torch.FloatTensor(state)
        state_normalized = (state_tensor - state_tensor.mean()) / (state_tensor.std() + 1e-6)

        logits = self.policy(state_normalized)
        logits = torch.clamp(logits, min=-20, max=20)
        concentration = torch.nn.functional.softplus(logits) + 1e-6
        concentration = torch.clamp(concentration, max=1e6)

        distribution = Dirichlet(concentration)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.detach().numpy(), log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
                           (discounted_rewards.std() + 1e-8)

        policy_loss = []
        for log_prob, G in zip(log_probs, discounted_rewards):
            policy_loss.append((-log_prob * G).unsqueeze(0))  # Critical fix here

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

    # ... rest of RLAgent remains the same ...

# Training function modified for tensor data
def train_agent(env,n_assets,episodes=500):
    agent = RLAgent(n_assets=n_assets)

    for episode in range(episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            action, log_prob = agent.get_action(state)
            reward, next_state, done = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        agent.update_policy(rewards, log_probs)
        if (episode+1) % 50 == 0:
            print(f'Episode {episode+1}, Total Return: {sum(rewards):.4f}')

    return agent

# Evaluation function for tensor data
def evaluate_agent(agent, env):
    weights = []
    state = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            action, _ = agent.get_action(state)
            weights.append(action)
            _, next_state, done = env.step(action)
            state = next_state

    return np.array(weights)


