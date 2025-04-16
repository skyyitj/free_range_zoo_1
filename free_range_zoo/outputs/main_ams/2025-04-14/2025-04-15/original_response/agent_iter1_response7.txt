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

    # Adjusted Temperature parameters, these are arbitrary and have been adjusted to get improved results
    level_temperature = 0.06
    intensity_temperature = 0.04
    distance_temperature = 0.02

    for task in range(num_tasks):

        # Calculate distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # Calculate score for each task using fire intensity, level, and distance
        # We are also considering the remaining suppressant here to penalize lower resources
        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature) +
            np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) -
            np.exp(fire_distance * distance_temperature)) * fire_putout_weight[task]

    # Return the index of the task with the highest score, this is what the agent will decide to take actions on
    max_score_task = np.argmax(scores)
    return max_score_task