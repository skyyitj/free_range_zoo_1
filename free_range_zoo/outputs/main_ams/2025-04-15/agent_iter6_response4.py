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

    total_fire_intensity = sum(fire_intensities)
    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Temperatures
    level_temperature = 0.35
    intensity_temperature = 0.25  # Increase sensitivity to fire intensity
    distance_temperature = 0.10

    for task in range(num_tasks):

        # Calculate Euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # Introduce a new variable, rate of fire extinguishing
        extinguish_rate = fire_intensities[task] / total_fire_intensity

        # Modify score calculation to consider extinguish_rate and relative fire intensity
        scores[task] = ((np.exp(-fire_levels[task] * level_temperature) +
                        can_put_out_fire * np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) -
                        fire_distance * np.exp(fire_distance * distance_temperature))
                        * extinguish_rate * np.sqrt(agent_suppressant_num)
                        * fire_putout_weight[task])

    # Return task index with maximum score
    max_score_task = np.argmax(scores)
    return max_score_task