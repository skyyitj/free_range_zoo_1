# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class ImprovedAgent(Agent):
    """Improved agent."""
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
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
        has_suppressant = self.observation['self'][:, 3] != 0
        self_x = self.observation['self'][0][1]
        self_y = self.observation['self'][0][0]
        for batch in range(self.parallel_envs):
            max_score = -100
            max_index = -1
            if len(self.argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)
                continue
            for index in range(self.argmax_store[batch].size(0)):
                if [index, 0] in action_space.spaces[0].enumerate():
                    fire_x = self.argmax_store[batch][index][1]
                    fire_y = self.argmax_store[batch][index][0]
                    distance = torch.sqrt(abs(fire_x - self_x) ** 2 + abs(fire_y - self_y) ** 2)
                    intensity = self.argmax_store[batch][index][2]
                    suppresant = self.observation['self'][batch][3]
                    beta = 0.1  
                    alpha = 1  
                    score = (2 / ((alpha * distance + 1) * (beta * intensity + 4)))
                    if score > max_score:
                        max_score = score
                        max_index = index
            if max_index == -1:  
                self.actions[batch].fill_(-1)
            else:
                self.actions[batch, 0] = max_index
            self.actions[batch, 1] = 0
        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.
        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        try:
            self.observation, self.t_mapping = observation
            self.t_mapping = self.t_mapping['agent_action_mapping']
        except Exception:  
            self.observation = observation
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]
        self.argmax_store = torch.zeros_like(self.fires)
        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]
