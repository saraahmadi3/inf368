import gym
import torch
import numpy as np
import multiprocessing as mp
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical



class AgentNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(AgentNet, self).__init__()
        self.affine = nn.Linear(num_inputs, 128)
        self.action_head = nn.Linear(128, num_outputs)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.affine(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

class PPOAgent:
    def __init__(self, env, config, policy_params):
        self.env = env
        self.config = config
        self.model = AgentNet(env.observation_space.shape[0], env.action_space.n)
        self.model.load_state_dict(policy_params)

    def step(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        next_state, reward, done, _ = self.env.step(action.item())
        return action.item(), log_prob, state_value, next_state, reward, done

    def update_policy(self, states, actions, old_log_probs, rewards, values, optimizer):
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(old_log_probs)
        rewards = torch.tensor(rewards)
        values = torch.cat(values)
        masks = torch.tensor([1.0] * len(rewards))

        # Adding last value for advantage calculation
        _, last_value = self.model(torch.from_numpy(states[-1]).float().unsqueeze(0))
        values = torch.cat([values, last_value.detach()])

        advantages = self.compute_advantages(rewards, masks, values)

        # Convert advantages to tensor and standardize
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert rewards to returns
        returns = advantages + values[:-1]

        # Optimization loop
        for _ in range(self.config['epochs']):
            idx = torch.randperm(len(states))
            for batch_indices in idx.split(self.config['batch_size']):
                sampled_states = torch.tensor(states)[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_old_log_probs = old_log_probs[batch_indices]
                sampled_advantages = advantages[batch_indices]

                # Forward pass
                new_probs, new_values = self.model(sampled_states)
                new_dist = Categorical(new_probs)
                new_log_probs = new_dist.log_prob(sampled_actions)

                # Calculating the ratio (pi_theta / pi_theta_old):
                ratio = torch.exp(new_log_probs - sampled_old_log_probs)

                # Clipped surrogate loss
                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * sampled_advantages
                loss = -torch.min(surr1, surr2).mean()  # Focus only on the clipping part

                # take gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
############################################################################################################

def worker(worker_id, policy_params, config, return_dict):
    """Worker process to collect data from the environment."""
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    env = gym.make(config['env_name'])
    agent = PPOAgent(env, config, policy_params)

    state = env.reset()
    rewards, log_probs, states, actions, values = [], [], [], [], []
    for _ in range(config['horizon']):
        action, log_prob, value, next_state, reward, done = agent.step(state)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        state = next_state if not done else env.reset()

    return_dict[worker_id] = {
        'states': states,
        'actions': actions,
        'log_probs': log_probs,
        'values': values,
        'rewards': rewards
    }

def main():
    config = {
        'env_name': 'CartPole-v1',
        'horizon': 2048,
        'learning_rate': 3e-4,
        'batch_size': 64,
        'epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'num_workers': 4
    }

    env = gym.make(config['env_name'])
    model = AgentNet(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    manager = mp.Manager()
    return_dict = manager.dict()

    for iteration in range(10):  # run for 10 iterations
        processes = []
        for i in range(config['num_workers']):
            p = mp.Process(target=worker, args=(i, model.state_dict(), config, return_dict))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        # Aggregate data from all workers
        aggregated_data = {k: [] for k in return_dict[0].keys()}
        for i in range(config['num_workers']):
            for key in aggregated_data.keys():
                aggregated_data[key].extend(return_dict[i][key])
        
        # Convert lists to tensors and perform PPO update
        states = torch.FloatTensor(aggregated_data['states'])
        actions = torch.LongTensor(aggregated_data['actions'])
        old_log_probs = torch.stack(aggregated_data['log_probs'])
        rewards = torch.FloatTensor(aggregated_data['rewards'])
        values = torch.stack(aggregated_data['values'])

        # Example PPO update, assuming `update_policy` is implemented
        agent = PPOAgent(env, config, model.state_dict())
        agent.update_policy(states, actions, old_log_probs, rewards, values, optimizer)

        print(f'Iteration {iteration + 1} complete.')

if __name__ == "__main__":
    main()



