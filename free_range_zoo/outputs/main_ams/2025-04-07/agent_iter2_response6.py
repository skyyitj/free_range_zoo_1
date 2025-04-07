# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class GenerateAgent:
    def __init__(self, agent_name, action_space, observation_space):
        self.agent_name = agent_name
        self.action_space = action_space
        self.observation_space = observation_space

    def observe(self, observation):
        # Convert observation into useful state: assume binary fire map exists
        # observation["fire_map"] shape: (H, W)
        self.fire_map = observation["fire_map"]
        self.agent_pos = tuple(observation["agent_pos"])  # (row, col)

    def act(self):
        # Simple greedy policy: move toward nearest fire
        if np.sum(self.fire_map) == 0:
            return 0  # No fire, stay put or no-op

        fire_coords = np.argwhere(self.fire_map > 0)
        min_dist = float('inf')
        best_move = 0  # Default: no-op

        # Map 0: stay, 1: up, 2: down, 3: left, 4: right
        moves = {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1)
        }

        for move_idx, delta in moves.items():
            new_pos = (self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1])
            for fire in fire_coords:
                dist = np.linalg.norm(np.array(new_pos) - fire)
                if dist < min_dist:
                    min_dist = dist
                    best_move = move_idx

        return best_move
