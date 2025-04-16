import numpy as np
from typing import List, Tuple
from scipy.spatial import distance

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
    fire_intensities: List[float],               # Current intensity vale of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    
    num_tasks = len(fire_levels)
    scores = -np.ones(num_tasks) * np.inf  # initialize scores with -inf 

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Adjusted temperature parameters
    level_temperature = 0.08
    intensity_temperature = 0.1
    distance_temperature = 0.15

    for task in range(num_tasks):

        # get euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # Skip the fire that the agent can't put out
        if fire_intensities[task] > can_put_out_fire:
            continue

        # calculate score for each task using fire intensity, level, and distance
        # all values are multiplied by suppressant_amount to penalize lower resources
        # 'fire_putout_weight' is directly applied as a multiplier to prioritize tasks
        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature) +
            np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) -
            np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task]

        # for the fire that can be put out, add a bonus according to the distance
        if fire_intensities[task] <= agent_fire_reduction_power:
            scores[task] += 1 / (1 + fire_distance)

    # return the index of the task with the highest score
    max_score_task = np.argmax(scores)
    return max_score_task