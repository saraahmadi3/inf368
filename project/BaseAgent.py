from abc import ABC, abstractmethod
from typing import Tuple, List


class AbstractAgent(ABC):
    """
    This is an abstract class for creating agents to interact with environments in Gymnasium.
    """

    def __init__(self):
        """
        Initialize the agent with the given action and observation space.
        
        Parameters:
        action_space: The action space of the environment.
        """
    
  
    @abstractmethod
    def select_action(self, observation):
        """
        Abstract method to define how the agent selects an action based on the current observation.
        
        Parameters:
        observation: The current observation from the environment.
        
        Returns:
        The action to be taken.
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        """
        Abstract method to define the learning process of the agent.
        """
        pass

    @abstractmethod
    def save(self, path:str = "") -> None:
        """
        Abstract method to save agent / model parameters.
        """
        pass

    @abstractmethod
    def load(self, path:str = "") -> None:
        """
        Abstract method to load agent / model parameters.

        Input: path: the path to the model parameters/weights
        """
        pass

