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
        has_suppressant = self.observation['self'][:, 3] > 0
        agent_pos = self.observation['self'][:, :2]  # [y, x]
        suppressant = self.observation['self'][:, 3]
        equipment_status = self.observation['self'][:, 2]
        fires = self.fire_info  # [batch, tasks, (y, x, intensity)]

        for batch in range(self.parallel_envs):
            best_score = -float('inf')
            best_index = -1
            agent_y, agent_x = agent_pos[batch]
            agent_supp = suppressant[batch]
            equipment_ok = equipment_status[batch] > 0

            if not equipment_ok or not has_suppressant[batch]:
                self.actions[batch].fill_(-1)
                continue

            for i in range(fires.size(1)):
                fy, fx, intensity = fires[batch, i]
                if intensity < 0:  # Masked/invalid entry
                    continue

                if [i, 0] not in action_space.spaces[0].enumerate():
                    continue

                # Normalize and weight scoring
                distance = torch.sqrt((fy - agent_y) ** 2 + (fx - agent_x) ** 2) + 1
                score = (intensity.item() + 1) / (distance.item())  # Basic priority: intensity/distance

                # Resource-aware scoring: reduce priority if low suppressant
                if agent_supp < 2:
                    score *= 0.5

                if score > best_score:
                    best_score = score
                    best_index = i

            if best_index == -1:
                self.actions[batch].fill_(-1)
            else:
                self.actions[batch, 0] = best_index
                self.actions[batch, 1] = 0

        self.actions[:, 1].masked_fill_(~has_suppressant, -1)
        return self.actions.tolist()

    def observe(self, observation: Dict[str, Any]) -> None:
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['agent_action_mapping']
        fire_obs = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]  # [y, x, intensity]
        self.fire_info = fire_obs.clone()
