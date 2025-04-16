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
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    num_tasks = len(fire_pos)
        
    scores = []
    for i in range(num_tasks):
        # Task Distance
        dist_y, dist_x = abs(fire_pos[i][0] - agent_pos[0]), abs(fire_pos[i][1] - agent_pos[1])
        task_dist = np.sqrt(dist_y**2 + dist_x**2)  # Euclidean distance to task
        
        # Task Intensity
        task_intensity = fire_intensities[i]
        
        # Fire Suppression Capability
        suppressant_capacity = agent_suppressant_num * agent_fire_reduction_power
        
        # Task Weight
        task_weight = fire_putout_weight[i]
        
        # Compute Score
        firefighting_capacity_score = suppressant_capacity / (task_intensity + 1e-10)
        distance_score = 1 / (task_dist + 1e-10)
        reward_score = task_weight
        
        # Apply transformation and compute final task score
        firefighting_capacity_temp = 1.0
        distance_temp = 2.0
        transformed_firefighting_capacity_score = np.exp(firefighting_capacity_score / firefighting_capacity_temp)
        transformed_distance_score = np.exp(distance_score / distance_temp)
        
        task_score = transformed_firefighting_capacity_score * transformed_distance_score * reward_score        
        scores.append(task_score)

    # Select the task with the maximum score.
    best_task = np.argmax(scores)
    
    return best_task