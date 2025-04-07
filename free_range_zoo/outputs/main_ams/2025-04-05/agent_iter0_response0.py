# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class WildFireAgent(Agent):

  def __init__(self, agent_name: str, parallel_envs: int) -> None:
    """
    Initialize the agent.

    Args:
        agent_name: str - Name of the subject agent
        parallel_envs: int - Number of parallel environments to operate on
    """
    self.agent_name = agent_name
    self.parallel_envs = parallel_envs
    self.agent_internal_state = None

  def act(self, action_space) -> List[List[int]] :
    """
    Return a list of actions, one for each parallel environment.

    Args:
      action_space: List[gymnasium.Space] - Current action space available to the agent.
    Returns:
        List[List[int]] - List of actions, one for each parallel environment.
    """
    actions = []
    
    agent_id = self.agent_name.split('_')[-1]

    for env_id in range(self.parallel_envs):
    
        fires = self.agent_internal_state['tasks'][env_id]
        agent_state = self.agent_internal_state['self'][env_id][agent_id]

        min_fire_intensity_index = torch.argmin(fires[:,2]).item() # index of the fire with minimum intensity
        
        if agent_state[2] == 1: # if fire power is at maximum, suppress the fire with minimum intensity
            actions.append([min_fire_intensity_index, agent_state[2]])
        else:
            actions.append([-1]) # else don't perform any action and let the agent power replenish

    return actions

  def observe(self, observation: Dict[str, Any]) -> None:
    """
    Observe the environment.

    Args:
        observation: Dict[str, Any] - Current observation from the environment.
    """
    self.agent_internal_state = observation
