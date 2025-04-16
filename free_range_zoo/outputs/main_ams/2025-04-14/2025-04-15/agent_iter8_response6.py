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
    fire_levels: List[int],                     # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    # Temperature variables
    weight_temperature = 0.5
    distance_temperature = 0.1 # Decreased the distance_temperature, as distance was driving the decision too much
    intensity_temperature = 0.3 # Increase the intensity_temperature, to focus more on high intensity fires

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # For each task, calculate a score. The score reflects the importance of the task
    for task in range(num_tasks):

        # calculate the euclidean distance between the fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        scores[task] = (
            fire_putout_weight[task] * np.exp(-fire_distance/distance_temperature) - 
            np.exp(fire_intensities[task]/intensity_temperature)
        )

    # Select the task with the maximum score
    selected_task = np.argmax(scores)
    return selected_task