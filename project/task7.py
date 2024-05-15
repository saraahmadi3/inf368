import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pickle
from itertools import product

# Define the ActorCritic class with separate actor and critic networks
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        value = self.critic(x)
        return mu, std, value

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, k_epochs=10, minibatch_size=64, gae_lambda=0.95):
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.minibatch_size = minibatch_size
        self.gae_lambda = gae_lambda
        self.actor_critic = ActorCritic(input_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        mu, std, _ = self.actor_critic(state)
        distribution = Normal(mu, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob

    def compute_gae(self, rewards, masks, values, next_value):
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, trajectory):
        states, actions, log_probs_old, returns, advantages = trajectory

        log_probs_old = torch.stack(log_probs_old).detach()
        states = torch.stack(states).detach()
        actions = torch.tensor(actions).detach()
        returns = torch.tensor(returns).unsqueeze(-1).detach()
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(self.k_epochs):
            indices = np.arange(states.shape[0])
            np.random.shuffle(indices)
            for start in range(0, states.shape[0], self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = indices[start:end]
                
                minibatch_states = states[minibatch_indices]
                minibatch_actions = actions[minibatch_indices]
                minibatch_log_probs_old = log_probs_old[minibatch_indices]
                minibatch_returns = returns[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]

                mu, std, values = self.actor_critic(minibatch_states)
                dist = Normal(mu, std)
                log_probs_new = dist.log_prob(minibatch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs_new - minibatch_log_probs_old)
                surr1 = ratios * minibatch_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * minibatch_advantages

                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                critic_loss = self.mse_loss(values, minibatch_returns)

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, env, total_episodes=1000, horizon=2048, batch_size=64):
        all_rewards = []

        states, actions, rewards, log_probs, values, masks = [], [], [], [], [], []

        for episode in range(total_episodes):
            state = env.reset()
            state = state[0] if isinstance(state, tuple) else state  # Extract state from tuple if necessary
            episode_rewards = 0

            for _ in range(horizon):  # Change from 1000 to horizon (2048 as per the hyperparameter table)
                action, log_prob = self.select_action(state)
                mu, std, value = self.actor_critic(torch.from_numpy(state).float())
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state[0] if isinstance(next_state, tuple) else next_state  # Extract next_state from tuple if necessary

                states.append(torch.from_numpy(state).float())
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value.item())
                masks.append(1 - done)

                state = next_state
                episode_rewards += reward

                if len(states) >= batch_size:
                    next_value = self.actor_critic(torch.from_numpy(state).float())[-1].item()
                    returns = self.compute_gae(rewards, masks, values, next_value)
                    advantages = [ret - val for ret, val in zip(returns, values)]
                    trajectory = (states, actions, log_probs, returns, advantages)
                    self.update(trajectory)

                    states, actions, rewards, log_probs, values, masks = [], [], [], [], [], []

                if done:
                    break

            all_rewards.append(episode_rewards)
            print(f"Episode {episode + 1}, Total Reward = {episode_rewards}")

        return all_rewards
    
    def evaluate(self, env, num_episodes=10):
        total_rewards = 0
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                state = state[0] if isinstance(state, tuple) else state
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state array to tensor
               
                mu, std, values = self.actor_critic(state_tensor)
                dist = Normal(mu, std)
                action = dist.sample()
               
                # Ensure action remains a 1-dimensional array with a single element
                action_numpy = action.squeeze().detach().numpy()  # Squeeze to potentially reduce dimensions
                if action_numpy.ndim == 0:  # If the result is a scalar, convert it back to an array
                    action_numpy = np.array([action_numpy])
                
                next_state, reward, done, _, _ = env.step(action_numpy)
                state = next_state
                episode_reward += reward

            total_rewards += episode_reward

        average_reward = total_rewards / num_episodes
        return average_reward

def run_experiments(env, param_grid):
    results = []

    for params in param_grid:
        lr, epsilon, k_epochs, batch_size = params
        print(f"Running experiment with lr={lr}, epsilon={epsilon}, k_epochs={k_epochs}, batch_size={batch_size}")

        ppo = PPOAgent(input_dim=4, action_dim=1, lr=lr, epsilon=epsilon, k_epochs=k_epochs, minibatch_size=batch_size)
        rewards = ppo.train(env, total_episodes=100)  # Reduce the number of episodes for quicker experimentation
        avg_reward = ppo.evaluate(env)
        
        result = {
            'lr': lr,
            'epsilon': epsilon,
            'k_epochs': k_epochs,
            'batch_size': batch_size,
            'average_reward': avg_reward,
            'rewards': rewards
        }
        results.append(result)

        with open('experiment_results_trial1.pkl', 'wb') as f:
            pickle.dump(results, f)

    return results

# Define the parameter grid
param_grid = list(product(
    [1e-4, 3e-4, 1e-3],  # Learning rates
    [0.1, 0.2, 0.3],     # Epsilon values
    [5, 10, 20],         # Epochs
    [32, 64, 128]        # Batch sizes
))

# Run experiments with the ActorCritic network architecture
env = gym.make('InvertedPendulum-v4')

print("Running experiments with the ActorCritic network architecture:")
results = run_experiments(env, param_grid)
