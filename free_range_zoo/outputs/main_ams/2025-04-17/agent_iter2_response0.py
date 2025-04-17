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

    optimal_task = -1
    highest_score = -np.inf

    for task_index in range(len(fire_pos)):
        # Calculate distance from agent to fire task
        distance = np.sqrt((agent_pos[0] - fire_pos[task_index][0]) ** 2 +
                           (agent_pos[1] - fire_pos[task_index][1]) ** 2)

        # Calculate normalized (0-1) scores for different aspects
        suppression_score = np.exp(0.1 * agent_fire_reduction_power)
        distance_score = np.exp(-0.2 * distance)
        intensity_score = np.exp(-0.05 * fire_intensities[task_index])
        level_score = np.exp(-0.07 * fire_levels[task_index])

        # Combine the scores with task priority weights to obtain the total score for the task
        total_score = (suppression_score + distance_score + intensity_score + level_score) * fire_putout_weight[task_index]

        if total_score > highest_score:
            highest_score = total_score
            optimal_task = task_index

    return optimal_task