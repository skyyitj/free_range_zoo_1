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
    num_tasks = len(fire_pos)        
    scores = []                     
    
    dist_temperature = 0.25
    level_temperature = 0.75
    intensity_temperature = 0.5
    weight_temperature = 0.5
    
    for i in range(num_tasks):
        dist = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        effect = agent_suppressant_num * agent_fire_reduction_power / max(fire_intensities[i], 1)
        
        # Count the amount of agents assigned to each fire
        num_agents_around_fire = len([1 for other_agent_pos in other_agents_pos if np.linalg.norm(np.array(other_agent_pos) - np.array(fire_pos[i])) < 2.0])
        # Slightly increase the influence of number of agents around a fire by including it in the exponential function
        agents_around_fire_temperature = 1.0
        agents_around_fire_weight = 0.5

        score = -np.exp(-dist/dist_temperature) \
                -np.exp(-(fire_levels[i]/level_temperature)) \
                -np.exp(-(fire_intensities[i]/intensity_temperature)) *np.exp(-num_agents_around_fire/agents_around_fire_temperature)\
                +np.exp(fire_putout_weight[i]/weight_temperature * num_agents_around_fire * agents_around_fire_weight)\
                +effect
                
        scores.append(score)
        
    return np.argmax(scores)