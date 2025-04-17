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
    distance_temp = 0.05  # Reduced emphasis on distance to allow broader task selection
    intensity_temp = 0.03  # Increased emphasis on fire intensity
    level_temp = 0.07  # Slightly increased emphasis on fire level
    resource_temp = 0.5  # New parameter for emphasizing the agent's remaining resources

    optimal_task = -1   
    highest_score = -np.inf  

    for task_index in range(len(fire_pos)):
        # Calculate distance from agent to fire task
        distance = np.sqrt((agent_pos[0] - fire_pos[task_index][0]) ** 2 +
                           (agent_pos[1] - fire_pos[task_index][1]) ** 2)

        # Calculate normalized (0-1) scores for each aspect
        distance_score = np.exp(-distance_temp * distance)
        intensity_score = np.exp(-intensity_temp * fire_intensities[task_index])
        level_score = np.exp(-level_temp * fire_levels[task_index])
        resource_score = np.exp(resource_temp * (agent_suppressant_num / 10))  # Encourage agents with more resources to take on tougher fires

        # Combine scores with task priority weights
        total_score = (distance_score + intensity_score + level_score + resource_score) * fire_putout_weight[task_index]

        # Select the task with the highest total score
        if total_score > highest_score:
            highest_score = total_score
            optimal_task = task_index

    return optimal_task