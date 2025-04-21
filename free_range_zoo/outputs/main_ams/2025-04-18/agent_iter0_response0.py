from typing import List, Tuple
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

    num_tasks = len(fire_pos)

    # Temperature parameters for score components transformation to control the sensitivity
    distance_temp = 0.1
    intensity_temp = 0.2
    weight_temp = 5.0

    best_score = float('-inf')
    task_index = 0

    for i in range(num_tasks):
        # Calculate the distance between agent and fire task
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)

        # Assuming agent's effectiveness decreases with increasing distance, normalize with exponential
        distance_score = np.exp(-distance * distance_temp)

        # Normalize fire intensity - lower intensity may be easier to handle, but we prioritize it based on the agent's capability
        # Assume fires with intensity higher than the agent's suppression power are urgent
        intensity_score = np.exp((fire_intensities[i] - agent_fire_reduction_power) * intensity_temp)

        # Weight based on input priority, higher weights should have higher priority
        weight_score = np.exp(fire_putout_weight[i] * weight_temp)

        # Calculate a combined score for each task
        # Incorporating agent's suppressant number to weigh in resource availability
        # NOTE: The score function is a simple example and can be modified for more sophisticated behavior.
        score = (distance_score + intensity_score + weight_score) * agent_suppressant_num

        # Update the best score and the corresponding task index
        if score > best_score:
            best_score = score
            task_index = i

    return task_index