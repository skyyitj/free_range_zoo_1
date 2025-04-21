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
    
    distance_temp = 0.3
    intensity_temp = 3.0
    weight_temp = 4.5
    suppressant_temp = 2.0

    task_scores = []
    for i in range(num_fires):
        distance = math.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        exp_distance = np.exp(-distance / distance_temp)
        
        exp_intensity = np.exp(fire_intensities[i] / intensity_temp)
        
        exp_weight = np.exp(fire_putout_weight[i] / weight_temp)

        # Calculate task score
        task_score = exp_distance * exp_intensity * exp_weight * np.exp(agent_suppressant_num / suppressant_temp)
        task_scores.append(task_score)

    # Return the index of the fire with maximum score
    return task_scores.index(max(task_scores))