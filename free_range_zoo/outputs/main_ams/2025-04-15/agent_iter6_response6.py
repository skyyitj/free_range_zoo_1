import numpy as np
from typing import List, Tuple
from scipy.spatial import distance

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

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Decrease the temperatures
    level_temperature = 0.10
    intensity_temperature = 0.05
    distance_temperature = 0.05

    for task in range(num_tasks):

        fire_distance = distance.euclidean(agent_pos, fire_pos[task])
        resource_efficiency = agent_suppressant_num/fire_intensities[task] if fire_intensities[task] != 0 else 1

        scores[task] = (
            agent_fire_reduction_power * np.exp(-fire_levels[task] * level_temperature) +
            can_put_out_fire * np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) -
            fire_distance * np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task] * resource_efficiency 

    max_score_task = np.argmax(scores)
    return max_score_task