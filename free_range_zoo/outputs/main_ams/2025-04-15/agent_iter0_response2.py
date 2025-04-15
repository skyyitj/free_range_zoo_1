import numpy as np
from typing import List, Tuple

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

    num_tasks = len(fire_pos)  # Number of fires
    task_scores = np.zeros(num_tasks)  # Initialize task scores

    # Parameters to balance different factors in task scoring
    distance_temperature = 1.0
    fire_level_temperature = 1.0
    fire_intensity_temperature = 1.0

    # Calculate Euclidean distance to all fires
    distances = [np.sqrt((fire_y - agent_pos[0]) ** 2 + (fire_x - agent_pos[1]) ** 2) for fire_y, fire_x in fire_pos]

    for task_id in range(num_tasks):
        # Calculate effective fire suppression. More effective suppression leads to better scoring
        effective_suppression = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensities[task_id])

        # Factor 1: Distance to fire (less is better)
        distance_score = np.exp(-distances[task_id] / distance_temperature)

        # Factor 2: Fire level (more is worse)
        fire_level_score = np.exp(-fire_levels[task_id] / fire_level_temperature)

        # Factor 3: Fire intensity (more is worse)
        fire_intensity_score = np.exp(-fire_intensities[task_id] / fire_intensity_temperature)

        # Factor 4: Fire suppression (more is better, scaled by task reward weight)
        suppression_score = fire_putout_weight[task_id] * effective_suppression

        # Final task score: Multiplicative integration of all factors
        task_scores[task_id] = distance_score * fire_level_score * fire_intensity_score * suppression_score

    # Choose task with highest score
    chosen_task = np.argmax(task_scores)
    return chosen_task