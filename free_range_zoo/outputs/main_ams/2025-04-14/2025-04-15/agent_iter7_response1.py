import numpy as np
from typing import List, Tuple
from scipy.spatial import distance

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Position of the agent (y, x)
    agent_fire_reduction_power: float,           # Fire suppression power
    agent_suppressant_num: float,                # Fire suppressant resources

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Positions of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value for each fire

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for suppression tasks
) -> int:

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Adjust temperatures
    level_temperature = 0.2 # Increasing the level temperature based on the feedback
    intensity_temperature = 0.1 # Reducing the intensity temperature
    distance_temperature = 0.25 # Increasing the distance temperature to prioritize closer fires

    for task in range(num_tasks):

        # Calculate Euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # Adjust the formulation to consider task priority factor and available suppressant amount in the score
        scores[task] = (
            np.exp(-fire_levels[task] / can_put_out_fire * level_temperature) +
            np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) -
            np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task] ** 2 * agent_suppressant_num

    # Return task index with maximum score
    max_score_task = np.argmax(scores)
    return max_score_task