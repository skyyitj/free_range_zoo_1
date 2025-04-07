# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

from free range_ zoo utils agent import Agent
class Agent:
    """Agent to fight the strongest available fire based on the observations."""
    
    def __init__(self) -> None:
        """Initialize the agent."""
    
    def act(self, observation: dict) -> int:
        """
        Return the optimal action based on the current state of the environment.
        
        Args:
            observation: dict - Current observations from the environment.
                Contains:
                - 'self' : [pos_x, pos_y, power, suppressant] or [pos_x, pos_y]
                - 'others' : [(pos_x, pos_y, power, suppressant), ...] or [(pos_x, pos_y), ...]
                - 'tasks' : [(y, x, level, intensity), ...]
                
        Returns:
            int - Action code indicating the optimal action.
                (e.g., Action code could be an index or encoded command for the agent to execute)
        """
        # Extract task observations
        tasks = observation['tasks']
        
        # Normalize factors to avoid saturation and to maintain relative importance
        intensity = [task[3] for task in tasks]  # Extract fire intensity
        
        # Find index of the strongest fire based on intensity
        strongest_fire_index = np.argmax(intensity)
        strongest_fire = tasks[strongest_fire_index]
        
        # Placeholder action code logic shows the selected fire's index; in a real scenario,
        # here we should calculate or retrieve the specific action to interact with the fire.
        return strongest_fire_index
    
    def observe(self, observation: dict) -> None:
        """
        Observe the environment. This is more of an educational or logging function in this context.
        
        Args:
            observation: dict - Current observation from the environment.
        """
        print("Observing:", observation)
