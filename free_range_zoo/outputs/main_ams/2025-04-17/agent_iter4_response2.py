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

    # Initialize the score and task index
    best_score = -np.inf  # Beginning with minus infinity to ensure any score is better
    best_task_index = -1

    # Iterate through all available fire tasks
    for i, (f_pos, f_level, f_intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        # Calculate euclidean distance from agent to fire
        distance = np.linalg.norm(np.array(agent_pos) - np.array(f_pos))

        # Calculate the potential efficiency of suppressing this fire, considering agent's resources and fire's intensity
        suppression_potential = min(agent_suppressant_num * agent_fire_reduction_power, f_intensity)
        efficiency = suppression_potential / (1 + distance)  # Adjusted with distance to favor closer fires

        # Incorporate fire putout weight to give priority to more valuable tasks
        score = efficiency * fire_putout_weight[i]

        # Update the best task and score, if current score is better
        if score > best_score:
            best_score = score
            best_task_index = i

    return best_task_index