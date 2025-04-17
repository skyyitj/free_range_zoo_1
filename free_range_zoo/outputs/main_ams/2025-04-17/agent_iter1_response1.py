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

    # Calculate effective distance that combines physical distance and effort needed
    distances = [
        np.sqrt((fpos[0] - agent_pos[0])**2 + (fpos[1] - agent_pos[1])**2) + fire_intensities[i] * 0.1
        for i, fpos in enumerate(fire_pos)
    ]

    # Evaluate how much each agent can reduce the fire taking into account their resources
    effectiveness_scores = [
        (agent_fire_reduction_power * agent_suppressant_num) / (fire_levels[i] + 1)
        for i in range(len(fire_levels))
    ]

    # Prioritize tasks based on a weight system that emphasizes on the rewards and effectiveness
    task_priority_scores = [
        fire_putout_weight[i] * effectiveness_scores[i] / (1 + distances[i])
        for i in range(len(distances))
    ]

    # Choose the task with the highest score
    best_task_index = np.argmax(task_priority_scores)
    return best_task_index