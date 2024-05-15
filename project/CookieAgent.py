import gymnasium as gym
import cookiedisaster
import numpy as np
# import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from BaseAgent import AbstractAgent
# print(torch.__version__)

SEED=2
np.random.seed(SEED)

# important parameters that need to be updated based on the environment
ENV_WIDTH=0.001
ENV_LIFETIME=1
MAX_TIME=ENV_LIFETIME*10*5 # 10 cookies elapse time (lifetime*steps_per_second*nr_of_elapsed_cookies)

# actor critic model
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)  

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

class CookieAgent(AbstractAgent):
    def __init__(self, input_dim=6, output_dim=3,epsilon=0, lr=0.001,save_path=None, MAX_TIME=MAX_TIME,ENV_LIFETIME=ENV_LIFETIME,ENV_WIDTH=ENV_WIDTH):
        super().__init__()
        self.policy = PolicyNetwork(input_dim, output_dim)
        self.value = ValueNetwork(input_dim)
        self.epsilon = epsilon
        self.MAX_TIME=MAX_TIME
        self.ENV_LIFETIME=ENV_LIFETIME
        self.ENV_WIDTH=ENV_WIDTH
        self.count=0
        if save_path:
            self.load(save_path)
        else:
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

    def select_action(self, observation):
        if self.count==0:
            print('First observation:',observation)
            self.update_env(observation)
            self.count+=1
        observation=self.preprocess_state(observation)
        state_tensor = torch.FloatTensor(observation).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        self.log_prob = distribution.log_prob(action)
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2])
        return action.item()

    def learn(self, state, reward, next_state, done):
        state=self.preprocess_state(state)
        next_state=self.preprocess_state(next_state)
        state_value = self.value(torch.FloatTensor(state).unsqueeze(0))
        next_state_value = self.value(torch.FloatTensor(next_state).unsqueeze(0))
        td_target = reward + (0.99 * next_state_value * (1 - int(done)))
        td_error = td_target - state_value
        
        # Critic loss
        critic_loss = td_error.pow(2)
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()

        # Actor loss
        actor_loss = -self.log_prob * td_error.detach()
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # update epsilon if the agent is learning
        if self.epsilon!=0:
            self.epsilon = max(0.01, self.epsilon * 0.995)
        

    def save(self, path):
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'value_state_dict': self.value.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])

    # helper functions for preprocessing the state

    def update_env(self,state):
        # global ENV_WIDTH,ENV_LIFETIME,MAX_TIME
        self.ENV_WIDTH=state['agent']['pos']*2
        self.ENV_LIFETIME=state['cookie']['time']
        self.MAX_TIME=self.ENV_LIFETIME*10*5
    
    def preprocess_state(self,state):
        # Assuming state is a dictionary like:
        # {'robot': {'pos': x, 'vel': y}, 'cookie': {'pos': z, 'time': w}}
        
        robot_pos = self.normalize(state['agent']['pos'], 0, self.ENV_WIDTH)
        robot_vel = self.normalize(state['agent']['vel'], -4, 4)
        cookie_pos = self.normalize(state['cookie']['pos'], 0, self.ENV_WIDTH)
        cookie_time = self.normalize(state['cookie']['time'], 0, self.ENV_LIFETIME)
        distance = robot_pos - cookie_pos
        direction = 1 if distance > 0 else -1
        distance = self.normalize(distance, -self.ENV_WIDTH, self.ENV_WIDTH)
        
        # Return the normalized state as a numpy array
        return np.array([robot_pos, robot_vel, cookie_pos, cookie_time,distance, direction])
    
    def normalize(self,value, min_value, max_value, scale_min=-1, scale_max=1):
        return ((value - min_value) / (max_value - min_value)) * (scale_max - scale_min) + scale_min


