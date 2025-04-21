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
    fire_putout_weight: List[float],             
) -> int:
    
    num_fires = len(fire_levels)
    scores = np.zeros(num_fires)

    for i in range(num_fires):
        fire = fire_pos[i]
        fire_distance = (abs(fire[0] - agent_pos[0]) + abs(fire[1] - agent_pos[1]))

        fire_intensity_temp = 10
        fire_distance_temp = 5
        fire_level_temp = 3

        # Score calculations based on fire intensity, agent distance from fire and fire level 
        fire_intensity_score = np.exp(-fire_intensities[i] / fire_intensity_temp)
        agent_distance_score = np.exp(-fire_distance / fire_distance_temp)
        fire_level_score = np.exp(-fire_levels[i] / fire_level_temp)
        
        # Combining all scores with their respective weights
        scores[i] = (fire_intensity_score * 0.3) + (agent_distance_score * 0.4) + (fire_level_score * 0.3)
        
        # Adjust score by fire weight and agent's available suppressant
        scores[i] *= fire_putout_weight[i] * min(1, agent_suppressant_num / (fire_intensities[i] + 1))

    return np.argmax(scores)