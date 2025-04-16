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
    intensity_temperature = 0.08
    distance_temperature = 0.04 

    for task in range(num_tasks):

        # get euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # calculate score for each task using fire intensity, level, and distance
        # all values are multiplied by suppressant_amount to penalize lower resources
        # 'fire_putout_weight' is directly applied as a multiplier to prioritize tasks
        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature) +
            can_put_out_fire * np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) -
            fire_distance * np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task]
        
    # apply a penalty score for fires that are reachable by other agents
    for agent in other_agents_pos:
        for task in range(num_tasks):
            agent_to_fire_distance = distance.euclidean(agent, fire_pos[task])
            if agent_to_fire_distance < fire_distance:
                scores[task] *= 0.5

    # return the index of the task with the highest score
    max_score_task = np.argmax(scores)
    return max_score_task

    # This policy function has made more changes to improve its performance.
    # It further reduces the temperature parameter for distance, encouraging agents to choose closer fires and increase efficiency.
    # A penalty score has been added for fires that are reachable by other agents which should help avoid multiple agents targeting the same fire and improve resource allocation.