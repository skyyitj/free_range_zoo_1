import numpy as np
from typing import Tuple, List

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
    best_task_index = -1
    highest_score = float('-inf')

    # Policy parameters
    distance_weight = 1.0
    intensity_weight = 5.0
    resource_conservation_weight = 2.0
    reward_factor = 2.0
    
    suppressant_factor = max(agent_suppressant_num, 1)  # avoid division by zero and encourage conservative use

    for i in range(num_tasks):
        y, x = fire_pos[i]
        distance = np.sqrt((agent_pos[0] - y)**2 + (agent_pos[1] - x)**2)
        effective_intensity = fire_levels[i] * fire_intensities[i]
        potential_reward = fire_putout_weight[i] * reward_factor

        # Composite score considering all factors
        score = (potential_reward / distance ** distance_weight) * (
            effective_intensity ** intensity_weight / suppressant_factor ** resource_conservation_weight)

        if score > highest_score:
            highest_score = score
            best_task_index = i

    return best_task_index