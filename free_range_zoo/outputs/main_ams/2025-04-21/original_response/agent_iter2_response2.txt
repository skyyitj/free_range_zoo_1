import math
import numpy as np
from typing import List, Tuple

def single_agent_policy(
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    num_fires = len(fire_pos)
    
    # Temperature variables for scoring transformations
    # Adjusted based on the policy evaluation feedback
    distance_temp = 0.1     # increased to prioritize nearby fires
    intensity_temp = 1.0    # keep unchanged to emphasize on high-intensity fires
    weight_temp = 2.0       # decreased to put more emphasis on high-priority fires 

    task_scores = []
    for i in range(num_fires):

        # Calculate Euclidean distance to the fire
        distance = math.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        # Normalize distance using exponential function
        exp_distance = np.exp(-distance / distance_temp)

        # Normalize fire intensity
        exp_intensity = np.exp(-fire_intensities[i] / intensity_temp)
        
        # Consider fire's priority weight
        exp_weight = np.exp(fire_putout_weight[i] / weight_temp)

        # Calculate task score considering balanced suppressant usage instead of maximum.
        suppressant_usage = min(agent_suppressant_num, (fire_intensities[i] / agent_fire_reduction_power))
        task_score = exp_distance * exp_intensity * exp_weight * suppressant_usage / agent_suppressant_num
        task_scores.append(task_score)

    # Return the index of the fire tasks with maximum score
    return task_scores.index(max(task_scores))