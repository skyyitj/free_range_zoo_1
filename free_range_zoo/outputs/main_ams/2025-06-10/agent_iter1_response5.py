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
    fire_putout_weight: List[float]
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    num_fires = len(fire_pos)
    best_index = 0
    highest_score = float('-inf')

    # Score settings
    distance_temperature = 0.5
    available_suppressant_temperature = 0.3
    fire_intensity_temperature = 0.1
    fire_level_temperature = 0.2
    
    for i in range(num_fires):
        firefighter_x, firefighter_y = agent_pos
        fire_x, fire_y = fire_pos[i]
        distance = np.sqrt((firefighter_x - fire_x)**2 + (firefighter_y - fire_y)**2)

        # Compute factors that play into the decision
        distance_score = np.exp(-distance * distance_temperature)
        suppressant_score = np.exp(agent_suppressant_num * available_suppressant_temperature)
        fire_intensity_score = np.exp(-fire_intensities[i] * fire_intensity_temperature)
        fire_level_score = np.exp(-fire_levels[i] * fire_level_temperature)
        weight_score = fire_putout_weight[i]

        # Compute the total score for choosing this fire
        score = (distance_score * suppressant_score * fire_intensity_score * fire_level_score * weight_score)
        
        # Find the fire task with the highest score
        if score > highest_score:
            highest_score = score
            best_index = i

    return best_index

# Example usage (for testing purposes, not part of the function itself):
# single_agent_policy(
#     (1.0, 1.0), 
#     10, 
#     1000,
#     [(0.5, 1.5), (2.0, -1.0)],
#     [(0, 0), (2, 2)],
#     [5, 8],
#     [50, 70],
#     [1.0, 1.5]
# )