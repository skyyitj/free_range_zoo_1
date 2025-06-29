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
    best_task_score = float('-inf')
    selected_task_index = -1

    # Adjusted parameters for better responsiveness
    distance_temp = 0.02  # Increase priority for nearby fires
    intensity_temp = 0.5  # Decrease focus a bit on high-intensity promoting efficiency
    suppressant_factor = 100.0  # Sharply emphasize suppressant conservation
    reward_boost = 2.5  # More weighting on higher reward tasks

    suppressant_potential = np.exp(-suppressant_factor * (1 - (agent_suppressant_num / 100)))

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        normalized_distance = np.exp(-distance_temp * distance)
        
        intensity_score = fire_levels[i] * (1 + fire_intensities[i])
        normalized_intensity = np.exp(-intensity_temp * intensity_score)

        # Revise score computation
        score = (fire_putout_weight[i] * reward_boost * normalized_distance + normalized_intensity) * suppressant_potential
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index