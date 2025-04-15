import numpy as np
from typing import List, Tuple
from scipy.spatial import distance

def single_agent_policy(
    agent_pos: Tuple[float, float],               # Current position of the agent (y, x)
    agent_fire_reduction_power: float,            # How much fire the agent can reduce
    agent_suppressant_num: float,                 # Amount of fire suppressant available

    other_agents_pos: List[Tuple[float, float]],  # Positions of all other agents [(y1, x1), (y2, x2), ...]

    fire_pos: List[Tuple[float, float]],          # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                       # Current intensity level of each fire
    fire_intensities: List[float],                # Current intensity value of each fire task

    fire_putout_weight: List[float],              # Priority weights for fire suppression tasks
) -> int:

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)
    task_threshold = 2 * agent_suppressant_num * agent_fire_reduction_power

    # Temperatures
    level_temperature = 0.35
    intensity_temperature = 0.15
    distance_temperature = 0.25
    fire_weight_temperature = 0.3

    for task in range(num_tasks):

        # Calculate Euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        if fire_levels[task] < task_threshold:
            scores[task] = (
                np.exp(-(fire_levels[task] / task_threshold) * level_temperature) +
                task_threshold * np.exp(-fire_intensities[task] / task_threshold * intensity_temperature) -
                fire_distance * np.exp(fire_distance * distance_temperature) +
                fire_putout_weight[task] * np.exp(fire_levels[task] / task_threshold * fire_weight_temperature)
            ) * np.sqrt(agent_suppressant_num)

        else:
            # Give a negative score to tasks that the agent can't finish
            scores[task] = -np.inf

    # Return task index with maximum score
    max_score_task = np.argmax(scores)
    return max_score_task