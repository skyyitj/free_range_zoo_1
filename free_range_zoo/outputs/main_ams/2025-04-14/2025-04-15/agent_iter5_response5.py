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
    scores = np.zeros(num_tasks)
    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power
    
    # Adjusted temperature parameters
    level_temperature = 0.25
    intensity_temperature = 0.15
    distance_temperature = 0.10
    power_temperature = 0.05 

    for task in range(num_tasks):

        # Get the Euclidean distance between agent and fire
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # Calculate score for each task using fire intensity, level, and distance from agent
        # The term 'suppressant_power' is added to consider the agent's current fire suppression capability
        suppressant_power = np.exp(-can_put_out_fire * power_temperature)
        
        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature) +
            suppressant_power * np.exp(-fire_intensities[task] * intensity_temperature) -
            fire_distance * np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task]

    # Select the task that has the highest score
    task_selected = np.argmax(scores)
    
    return task_selected