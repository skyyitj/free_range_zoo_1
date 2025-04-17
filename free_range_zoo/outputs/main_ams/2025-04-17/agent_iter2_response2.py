from typing import Tuple, List
import numpy as np

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
    """
    Choose the optimal fire-fighting task for a single agent.
    """

    # Adjusted parameters for scoring transformations based on analysis
    distance_temp = 0.05  
    intensity_temp = 0.1  
    level_temp = 0.1  

    highest_score = -np.inf  
    optimal_task = -1   

    for task_index in range(len(fire_pos)):
        # Calculate distance from agent to the fire task
        distance = np.sqrt((agent_pos[0] - fire_pos[task_index][0])**2 + (agent_pos[1] - fire_pos[task_index][1])**2)

        # Calculate scores for each aspect
        distance_score = np.exp(-distance_temp * distance)
        intensity_score = np.exp(-intensity_temp * fire_intensities[task_index])
        level_score = np.exp(-level_temp * fire_levels[task_index])

        # Combine scores with task priority weights
        total_score = (distance_score + intensity_score + level_score) * fire_putout_weight[task_index]

        # Select the task with the highest total score
        if total_score > highest_score:
            highest_score = total_score
            optimal_task = task_index

    return optimal_task