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
    fire_levels: List[int],                    # Current intensity level of each fire task
    fire_intensities: List[float], 


    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    # Define the number of tasks
    num_tasks = len(fire_pos)
    
    task_scores = []
    for task in range(num_tasks):
        # Compute the distance from the agent to task
        dist = np.sqrt((agent_pos[0] - fire_pos[task][0]) ** 2 + (agent_pos[1] - fire_pos[task][1]) ** 2)
        # Estimate remaining fire after the agent's action
        remaining_fire = fire_levels[task] - agent_fire_reduction_power * agent_suppressant_num
        # Stage 1 score calculation: High remaining fire ==> low score
        score_remaining_fire = 1 / (1+ remaining_fire)
        # Stage 2 score calculation: Distance factor - far away tasks get less score
        score_distance = 1 / (1 + dist)
        # Combine scores and normalize with exp function
        combined_score = np.exp(score_remaining_fire / 0.1) * np.exp(score_distance / 10.0) * fire_putout_weight[task]
        task_scores.append(combined_score)

    # Return task with highest score
    return int(np.argmax(task_scores))