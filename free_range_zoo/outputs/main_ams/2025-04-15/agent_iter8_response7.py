import numpy as np
from typing import List, Tuple
from scipy.spatial import distance

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],             # Current position of the agent (y, x)
    agent_fire_reduction_power: float,          # How much fire the agent can reduce
    agent_suppressant_num: float,               # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]],# Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],        # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                     # Current intensity level of each fire
    fire_intensities: List[float],              # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],            # Priority weights for fire suppression tasks
) -> int:

    num_tasks = len(fire_pos)
    scores = np.zeros(num_tasks)
    num_agents = len(other_agents_pos) + 1 # Adding 1 for the current agent

    can_put_out_fire = agent_fire_reduction_power * agent_suppressant_num

    # Temperatures
    level_temp = 0.5     # Increase to prioritize fires that can be put out completely
    intensity_temp = 0.2 # Decrease to prioritize closer fires
    distance_temp = 0.4  # Balance between distance and fire levels

    for task in range(num_tasks):

        # calculate the euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task]) 

        # Distance score components
        agent_distance_contribution = np.exp(-fire_distance * distance_temp)
        other_agents_distance_contribution = np.sum([np.exp(-distance.euclidean(other_agent_pos, fire_pos[task]) * distance_temp) for other_agent_pos in other_agents_pos])/(num_agents - 1)

        # Task score components
        level_contribution = np.exp(-(fire_levels[task] / (can_put_out_fire + 1e-10)) * level_temp)
        intensity_contribution = np.exp(-fire_intensities[task] * intensity_temp) 
        reward_contribution = fire_putout_weight[task]

        # Score Calculation
        scores[task] = reward_contribution * (agent_distance_contribution - other_agents_distance_contribution + level_contribution + intensity_contribution) 

    # Return task index with maximum score
    return int(np.argmax(scores))