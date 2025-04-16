import numpy as np
from typing import List, Tuple

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], 

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                      
    fire_intensities: List[float],               

    # === Task Prioritization ===
    fire_putout_weight: List[float],             
) -> int:
    
    num_tasks = len(fire_pos)                    
    scores = []                                  
    
    # Change temperature parameters
    dist_temperature = 2.0
    level_temperature = 2.0
    intensity_temperature = 0.5
    weight_temperature = 0.5
    
    # Iterate over every task
    for i in range(num_tasks):
        # Distance to fire
        dist = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        # Potential effects of agent's suppressant on the fire intensity
        effect = agent_suppressant_num * agent_fire_reduction_power / max(fire_intensities[i], 1.0)
        # Score based on distance, fire level, intensity, task weight, and potential effect
        score = -np.exp(-dist/dist_temperature) \
                -np.exp(-fire_levels[i]/level_temperature) \
                -np.exp(-fire_intensities[i]/intensity_temperature) \
                +np.exp(fire_putout_weight[i]/weight_temperature) \
                +np.exp(effect/intensity_temperature)
        scores.append(score)
    
    # Return the index of task with maximum score
    # Since scores are negative, argmin() is used
    return np.argmin(scores)