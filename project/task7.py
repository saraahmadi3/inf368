import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import gymnasium


env = gymnasium.make('InvertedPendulum-v4')



# Actor-Critic Network for Continuous Action Spaces
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log standard deviation
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean)  # Standard deviation
        dist = torch.distributions.Normal(mean, std)
        return dist, self.critic(x)

# PPO Agent with Clipping Method
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, clip_epsilon=0.2):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99
        self.clip_epsilon = clip_epsilon

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = self.model(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def train(self, states, actions, rewards, next_states, dones, old_log_probs):
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(next_states)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)

        _, values = self.model(states_tensor)
        dists, next_values = self.model(next_states_tensor)

        new_log_probs = dists.log_prob(actions_tensor).sum(axis=1, keepdim=True)

        advantages = rewards_tensor + self.gamma * next_values.squeeze() * (1 - dones_tensor) - values.squeeze()

        ratios = (new_log_probs - old_log_probs_tensor).exp()
        clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        critic_loss = advantages.pow(2).mean()
        loss = surrogate_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training Function
def train_agent(agent, env, episodes=1000):
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        states, actions, rewards, next_states, dones, old_log_probs = [], [], [], [], [], []

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _,_ = env.step([action])  # Action needs to be a list
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            old_log_probs.append(log_prob)
            state = next_state
            total_reward += reward

        agent.train(states, actions, rewards, next_states, dones, old_log_probs)
        print(f'Episode {episode + 1}: Total Reward: {total_reward}')

env = gymnasium.make('InvertedPendulum-v4')
agent = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
train_agent(agent, env, episodes=1000)
