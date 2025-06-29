import numpy as np
from typing import Tuple, List, float


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

    # Adjusted parameters
    distance_weight = 2  # Encourages closer targets, reduced from previous value.
    intensity_weight = 3  # Places more emphasis on tackling high-intensity fires.
    reward_weight = 5  # Strong emphasis on higher reward weights.
    suppressant_weight = 8  # High value to ensure better usage of suppressants.

    # Rebalancing the score to enhance agent performance on crucial metrics
    for i in range(num_tasks):
        # Distance score modification
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        distance_score = np.exp(-distance_weight * distance)
        
        # Intensity score modification
        intensity = fire_intensities[i] * fire_levels[i]
        intensity_score = np.exp(intensity_weight * intensity)
        
        # Reward score modification
        reward_score = fire_putout_weight[i] ** reward_weight
        
        # Suppressant score modification
        available_suppressant = agent_suppressant_num / max(1, intensity)
        suppressant_score = np.exp(suppressant_weight * available_suppressant)

        # Calculate score 
        score = distance_score * intensity_score * reward_score * suppressant_score

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index