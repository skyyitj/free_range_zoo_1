from typing import Tuple, List
import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.

    Returns:
        int: Index of the selected fire task (0 to num_tasks-1)
    """
    num_tasks = len(fire_pos)
    agent_y, agent_x = agent_pos

    # Parameters for scoring transformations
    distance_temperature = 1.0  # Temperature for distance transformation
    fire_intensity_temperature = 1.0  # Temperature for fire intensity transformation

    best_task_index = -1
    best_task_score = -float('inf')  # Initialize with negative infinity

    for task_idx in range(num_tasks):
        # Extract fire information
        fire_y, fire_x = fire_pos[task_idx]
        fire_intensity = fire_intensities[task_idx]
        fire_level = fire_levels[task_idx]
        fire_weight = fire_putout_weight[task_idx]

        # Compute distance-based score (closer fires are prioritized)
        distance = np.sqrt((fire_y - agent_y) ** 2 + (fire_x - agent_x) ** 2)
        distance_score = np.exp(-distance / distance_temperature)

        # Normalize fire intensity score (higher intensity = higher priority)
        intensity_score = np.exp(fire_intensity / fire_intensity_temperature)

        # Compute resource-based score (consider suppressant and reduction power)
        max_suppressable_fire = agent_suppressant_num * agent_fire_reduction_power
        resource_based_score = min(fire_intensity, max_suppressable_fire) / (fire_intensity + 1e-6)

        # Compute overall score for the task
        task_score = fire_weight * (distance_score + intensity_score + resource_based_score)

        # Update the best task if the current task has a higher score
        if task_score > best_task_score:
            best_task_score = task_score
            best_task_index = task_idx

    return best_task_index