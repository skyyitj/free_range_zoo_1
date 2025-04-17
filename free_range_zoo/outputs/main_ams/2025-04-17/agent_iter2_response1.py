from typing import List, Tuple
import numpy as np

def single_agent_policy(
    # Agent Properties
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    # Team Information
    other_agents_pos: List[Tuple[float, float]],

    # Fire Task Information
    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                    
    fire_intensities: List[float],              

    # Task Prioritization
    fire_putout_weight: List[float],             
) -> int:
    """
    This function selects the optimal fire-fighting task for a single agent.
    """
    # Define a placeholder for the objective value of the best task found
    best_task_value = -np.inf  
    # Define a placeholder for the index of the best task found
    best_task_idx = -1  

    # Iterate over all the tasks
    for task_idx in range(len(fire_pos)):
        # Calculate the distance to the fire
        distance_to_fire = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[task_idx]))
        
        # Calculate potential impact of agent on the fire
        potential_impact = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensities[task_idx])

        # Define the importance of a task based on its weight and current fire intensity
        task_importance = fire_putout_weight[task_idx] * fire_intensities[task_idx]

        # Calculate the value of the current task for the agent
        task_value = (potential_impact + task_importance) / (distance_to_fire + 1)  
        
        # If the current task value is greater than the best found so far
        if task_value > best_task_value:
            # Update the best task value and index
            best_task_value = task_value
            best_task_idx = task_idx

    # Return the index of the best task
    return best_task_idx