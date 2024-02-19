import numpy as np
import os
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


class MountainLiteEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        
        
        self.MAP = np.genfromtxt(os.path.join('mountain','envs','the_hill2.txt'),float)
        self.height = self.MAP.shape[0]  # The height of the grid
        self.width = self.MAP.shape[1]  # The width of the grid
        self._cummlative_reward = 0 

        self._step = 0
        #self.graphics[15,0,:] = (173, 216, 230,0.1) #start

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Dict(
                        {"pos": spaces.Box(
                            low=np.array([0, 0]), 
                            high=np.array([self.width - 1, self.height - 1]), 
                            dtype=np.int32
                        )
                          }
                ) 
            }
        )

        # We have 3 actions, corresponding to "leftforward", "straigth", "left", "rightforward"
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.action_to_direction = {
            0: np.array([-1, -1]), #backward left
            1: np.array([-1, 0]), #left
            2: np.array([-1, 1]), #forward left
            3: np.array([1, -1]), #backward right
            4: np.array([1, 0]), # right
            5: np.array([1, 1]),  #forward right
            6: np.array([0, -1]), #backward
            7: np.array([0, 1]),  #forward
        }
       

        

        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    

    def _get_obs(self):
        return {"agent": {"pos": self._agent_location}}


    def _get_info(self):
        return {"distance": abs(self._agent_location[1] - self._target_location[1]), "steps": self._step}
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = np.array([15, 0])
        self._step = 0
        self._cummlative_reward = 0

        self._target_location = np.array([15,99])
        

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    
    def _transition(self, action):
        r = np.random.uniform()
        if action in [0,3,6]: #backwards, 100%
            return self.action_to_direction[action]
        if  action in [1,4]: #sideways
            if r<1/16:
                return self.action_to_direction[action-1] # sideways backward
            elif r<1/8:
                return self.action_to_direction[action+1] # sideways forward
            else:
                return self.action_to_direction[action] # sideways 

        if  action in [3,5]: #forward sideways
            if r<1/16:
                return self.action_to_direction[action-1] # sideways 
            elif r<1/8:
                return self.action_to_direction[7] # forward
            else:
                return self.action_to_direction[action] # sideways forward
            
        return self.action_to_direction[action]


    
    def step(self, action):
        # Map the action (element of {0,1,2}) to the direction to move in
        
        direction = self._transition(action)
        
        # We use `np.clip` to make sure we don't leave the grid
        next_pos = self._agent_location + direction
        next_pos[0] = np.clip(self._agent_location[0] + direction[0],a_min=0, a_max=30)
        next_pos[1] = np.clip(self._agent_location[1] + direction[1],a_min=0, a_max=99)
    

        self._step += 1    
        reward = -(self.MAP[self._agent_location[0], self._agent_location[1]])
        
        self._agent_location = next_pos
        
        
        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)# or (self._step >=10000)
        if terminated:
            reward =0
        self._cummlative_reward +=reward
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def render(self):
        return None

    def close(self):
        return None
        