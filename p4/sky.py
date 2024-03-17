
import gymnasium

import numpy as np

import random
import skyscraper
env = gymnasium.make('skyscraper/GridWorld-v0')
import matplotlib.pyplot as plt

data = np.loadtxt('powered_flight.txt', dtype=int)
data[:,:3] -= 1
data[:,4:6] -= 1

transition_function = {}  # Transition function: (state, action) -> next state

for i, j, a, r, i_prime, j_prime in data:
    
    current_state = (i, j)
    next_state = (i_prime, j_prime)
    action = a
    
    transition_function[(current_state, action)] = next_state
transition_function

reward_function = {}  # Reward function: (state, action, next state) -> reward

for i, j, a, r, i_prime, j_prime in data:
    current_state = (i, j)
    next_state = (i_prime, j_prime)
    action = a
    
    reward_function[(current_state, action)] = r
    
    
reward_function

def model_M(current_state, action, transition_function, reward_function):
   
    next_state = transition_function.get((current_state, action), None)  # Get the next state from T
    
    reward = reward_function.get((current_state, action), 0)  # Get the reward from R
    
    return next_state, reward

# print(model_M((12, 17), 0, transition_function, reward_function))

action_space = [0, 1]  
alpha = 0.4  #originally 0.5
gamma = 0.99
epsilon = 0.3
n_planning_steps = 1000

T = 1000  # Total number of real interactions
state = env.reset()[0]['agent']['pos']
state = tuple(state)

Q=np.zeros((env.height,env.width,env.action_space.n))


# Choose action based on ε-greedy policy
def choose_action(state, Q, epsilon):
    if random.random() < epsilon:
        
        return env.action_space.sample()  
    else:
        
        return np.argmax(Q[state])  


def q_learning(state, action, reward, next_state, value, discound,lr):

    i,j = state
    prev_value = value[i,j,action]

    if next_state is None or np.isnan(next_state).any():
        max_value = 0
    else:
        n_i,n_j = next_state
        max_value = np.max(value[n_i,n_j, :])

    delta = reward + discound * max_value - prev_value

    value[i,j,action] = prev_value + lr * delta

    return value
    

def print_optimal_policy(Q):
   return np.argmax(Q,axis=2)

print_optimal_policy(Q)


# Main loop for Dyna-Q
for k in range(T):
    
    if k % 100 == 0:
        print(f"Real interaction {k}/{T}")
    
    state =env.reset()[0]['agent']['pos']
    state = tuple(state)
    start_state=state

    done = False
    while not done:
        
        
        action = choose_action(state, Q, epsilon)
        
        next_state, reward, done, err, info = env.step(action) 
        
        next_state = tuple(map(int,next_state['agent']['pos']))
        
        q_learning(state, action, reward, next_state, Q, gamma, alpha)
        transition_function[(state, action)] = next_state
        reward_function[(state, action)] = reward
        
        crashed  = (start_state==next_state and state != start_state)
        
        state = next_state
        
        if done or crashed:
            done = True


    for m in range(n_planning_steps):
    
        sampled_state, sampled_action = random.choice(list(transition_function.keys()))
    
        if (sampled_state, sampled_action) in transition_function:
            simulated_next_state = transition_function[(sampled_state, sampled_action)]
            simulated_reward = reward_function[(sampled_state, sampled_action)]
            q_learning(sampled_state, sampled_action, simulated_reward, simulated_next_state,Q, gamma,alpha)

policy=print_optimal_policy(Q)
policy

#this function is used to run the policy
def run_with_policy(env, policy_matrix, num_steps=1000):
    observation, _ = env.reset()
    current_pos = tuple(map(int, observation["agent"]["pos"]))

    for step in range(num_steps):
        if step % 10 == 0:
            print(f"Steps: {step} to {step + 99} of {num_steps}")

        action = policy_matrix[current_pos]
        print("action: ",action)
       
        observation, reward, done, _, _ = env.step(action)

        current_pos = tuple(map(int, observation["agent"]["pos"]))
        
        print(f"Current Position: {current_pos}")

        if done:
            observation, _ = env.reset()
            current_pos = tuple(map(int, observation["agent"]["pos"]))

env = gymnasium.make('skyscraper/GridWorld-v0',render_mode="human")
print("Run with policy")
#run_with_policy(env, policy)


# Assuming env is already defined and initialized somewhere in your code
env.reset()
env_map = env.MAP  # Make sure this is the correct way to access the map

# Visualize the environment
plt.imshow(env_map, cmap='copper', interpolation='nearest')
plt.colorbar()
plt.show()

def follow_policy(env, policy):
    total_reward = 0
    trajectory = []
    done = False
    state, info = env.reset()
    state = tuple(state['agent']['pos'])
    trajectory.append(state)
    
    while not done:
        action = policy[state[0], state[1]]  # Adjust based on your state definition
        next_state, reward, done, _, _ = env.step(action)
        next_state = tuple(map(int, next_state['agent']['pos']))
        total_reward += reward
        state = next_state
        trajectory.append(state)
        
        if done:
            break  # Exit the loop if the episode is done
    
    return total_reward, trajectory

# Convert Q-values to policy
policy = np.argmax(Q, axis=2)  # Make sure this aligns with how your Q-table is structured

# Follow the policy
control_total_reward, control_trajectory = follow_policy(env, policy)
print("Total Reward:", control_total_reward)
print("Trajectory:", control_trajectory)

plt.imshow(env_map, cmap='copper', interpolation='nearest')  # Use 'viridis' or another colormap as needed
plt.colorbar()
plt.plot([x[1] for x in control_trajectory], [x[0] for x in control_trajectory], color='purple', markersize=5, linewidth=2)

# Show the plot
plt.show()





import gymnasium
import numpy as np
import random
import matplotlib.pyplot as plt
import skyscraper
# Load data and adjust indexing to match Python's 0-based indexing
data = np.loadtxt('powered_flight.txt', dtype=int)
data[:,:3] -= 1
data[:,4:6] -= 1
action_space = [0, 1]  
ls = 0.4  #originally 0.5
discount = 0.99
epsilon = 0.3
# Initialize environment
env = gymnasium.make('skyscraper/GridWorld-v0')

# Define the transition and reward functions based on the loaded data
transition_function = {}
reward_function = {}
for i, j, a, r, i_prime, j_prime in data:
    current_state, next_state = (i, j), (i_prime, j_prime)
    transition_function[(current_state, a)] = next_state
    reward_function[(current_state, a, next_state)] = r

# Define the model function for the environment
def model_M(current_state, action):
    next_state = transition_function.get((current_state, action), current_state)  # Default to current state if not found
    reward = reward_function.get((current_state, action, next_state), 0)  # Default reward is 0
    return next_state, reward

# Dyna-Q algorithm function

def run_dyna_q(env, n_planning_steps, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=100):
    Q = np.zeros((env.height, env.width, 2))  # Initialize Q-values
    for episode in range(episodes):
        # Reset the environment for a new episode
        state = tuple(env.reset()[0]['agent']['pos'])
        done = False
        while not done:
            action = choose_action(state, Q, epsilon)  # Choose an action based on the current policy
            next_state, reward, done, _, _ = env.step(action)  # Take the action in the environment
            next_state = tuple(map(int,next_state['agent']['pos']))  # Get the next state
            # Update Q-values using the Q-learning algorithm
            Q = q_learning(state, action, reward, next_state, Q, gamma, alpha)
            state = next_state  # Move to the next state

            # Planning: randomly sample previous experiences to update Q-values
            for _ in range(n_planning_steps):
                sampled_state, sampled_action = random.choice(list(transition_function.keys()))
                simulated_next_state, simulated_reward = model_M(sampled_state, sampled_action)
                Q = q_learning(sampled_state, sampled_action, simulated_reward, simulated_next_state, Q, gamma, alpha)
    return np.argmax(Q, axis=2)  # Return the derived policy

# Implementation for q_learning and choose_action remains the same

# Define Q-learning function
def q_learning(state, action, reward, next_state, value, discount, lr):
    i,j = state
    prev_value = value[i,j,action]

    if next_state is None or np.isnan(next_state).any():
        max_value = 0
    else:
        n_i,n_j = next_state
        max_value = np.max(value[n_i,n_j, :])

    delta = reward + discount * max_value - prev_value

    value[i,j,action] = prev_value + lr * delta

    return value

# Define action choice function
def choose_action(state, Q, epsilon):
    # Implementation of ε-greedy policy
    # (This needs to be provided based on your Dyna-Q setup)
    if random.random() < epsilon:
        
        return env.action_space.sample()  
    else:
        
        return np.argmax(Q[state]) 
    
# Run Dyna-Q for different planning steps and collect policies
planning_steps_list = [0, 5, 10, 20, 50, 100]
policies = {}
for n_planning_steps in planning_steps_list:
    policies[n_planning_steps] = run_dyna_q(env, n_planning_steps)

# Evaluate and compare the policies
# (Evaluation function needs to be defined based on your environment and requirements)

# Example of evaluating and plotting could go here

