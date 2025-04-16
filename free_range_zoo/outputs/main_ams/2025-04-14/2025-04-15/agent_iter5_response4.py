from typing import List, Tuple
import numpy as np
from scipy.spatial import distance

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

    # Calculate the number of tasks and initialize the scores
    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    # Calculate the quantity of fire a agent can putout
    can_putout_fire = agent_suppressant_num * agent_fire_reduction_power

    # Increased temperature parameters
    level_temperature = 0.45
    intensity_temperature = 0.2
    distance_temperature = 0.1

    for task in range(num_tasks):

        # calculate euclidean distance between agent and fire
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # calculate scores by balancing between fire level, intensity, distance, and availability of fire suppressants
        scores[task] = (
            np.exp(-fire_levels[task] / can_putout_fire * level_temperature) + 
            np.exp(-fire_intensities[task] / can_putout_fire * intensity_temperature) -
            np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task]

    # Return the task with maximum score
    return np.argmax(scores)