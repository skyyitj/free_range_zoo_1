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
    
    scores = []
    for task_index in range(len(fire_pos)):
        # Calculate distance 
        distance = np.sqrt((fire_pos[task_index][0] - agent_pos[0])**2 + (fire_pos[task_index][1] - agent_pos[1])**2)
        
        # Expected contribution of the agent towards putting out the fire
        expected_contribution = agent_fire_reduction_power / (fire_intensities[task_index] + 1)
        
        # Calculate effective weight considering the urgency and importance of the fire
        effective_weight = fire_putout_weight[task_index] * (1 + 1 / (fire_levels[task_index] + 1))

        # Calculate a score considering distance, expected contribution, and fire level importance
        score = (expected_contribution + effective_weight) / (distance + 1)
        scores.append(score)
    
    # Choose the fire task with the highest score
    best_task_index = np.argmax(scores)
    return best_task_index