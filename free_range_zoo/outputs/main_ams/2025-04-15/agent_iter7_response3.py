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

    # Temperatures for each component
    level_temperature = 0.40
    intensity_temperature = 0.20
    distance_temperature = 0.15 # Decreasing temperature to reduce the impact from distance to fire

    for task in range(num_tasks):

        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # Adjusting the calculation to increase the impact from fire level and fire intensity
        scores[task] = (
            np.exp(-(fire_levels[task] / can_put_out_fire) * level_temperature) +
            can_put_out_fire * np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) -
            fire_distance * np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task] * np.sqrt(agent_suppressant_num)

    max_score_task = np.argmax(scores)
    return max_score_task