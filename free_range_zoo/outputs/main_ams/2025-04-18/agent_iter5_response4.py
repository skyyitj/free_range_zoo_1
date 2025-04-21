import numpy as np
from typing import List, Tuple

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
  
    other_agents_pos: List[Tuple[float, float]], 
     
    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float], 
    
    fire_putout_weight: List[float]
) -> int:

    max_score = float('-inf')
    best_fire = None
    temp_suppression = 0.1
    temp_dist = 0.05
    temp_fire_level = 0.2

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        dist = np.sqrt((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        suppression_score = np.exp((np.log(fire_weight + 1) *  suppression_power) / temp_suppression)
        dist_score = np.exp(-dist / temp_dist)
        fire_level_score = np.exp(fire_level/temp_fire_level)

        score = fire_weight * suppression_score * dist_score * fire_level_score

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire