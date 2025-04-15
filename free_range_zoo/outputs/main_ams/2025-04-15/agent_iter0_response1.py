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
    fire_levels: List[int],                      # Current intensity level of each fire task
    fire_intensities: List[float],               # Current intensity value of each fire task
 
    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
    ):
    
    # Embedded function to calculate euclidean distance 
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    # Variables for calculation scores
    max_dist = np.sqrt(2 * (100**2))  # assuming 100x100 grid
    distance_temperature = 1 / max_dist
    intensity_temperature = 1

    # Calculate score for each task based on distance, intensity and priority
    task_scores = []
    for i in range(len(fire_pos)):
        
        # Task selection score components
        dist_score = np.exp(-euclidean_distance(agent_pos, fire_pos[i]) * distance_temperature)
        intensity_score = np.exp(-fire_intensities[i] * intensity_temperature)
        priority_score = fire_putout_weight[i]
        
        # Total task score
        task_score = dist_score * intensity_score * priority_score

        task_scores.append(task_score)
    
    # Choose task with the highest score
    chosen_task = np.argmax(task_scores)

    return chosen_task-b