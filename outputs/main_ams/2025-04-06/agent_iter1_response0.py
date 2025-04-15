# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class GenerateAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.
        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        # Observations are now correctly extracted
        fire_power = self.observation['self'][2]  # fire power available
        suppressant = self.observation['self'][3]  # suppressant available
        fire_level = self.observation['tasks'][2]  # fire level in current task
        intensity = self.observation['tasks'][3]  # intensity of fire in current task
        
        # Decision-making logic
        actions = []
        for i in range(self.parallel_envs):
            if intensity[i] > 0:  # if there's fire to fight
                if fire_power[i] > 0 and suppressant[i] > 0:
                    actions.append([random.choice(range(len(action_space)))]  # Fight action
                else:
                    actions.append([random.choice([-1, 0])])  # No-op or fallback action
            else:
                actions.append([random.choice([0, -1])])  # No-op or stand by action
        
        return actions
