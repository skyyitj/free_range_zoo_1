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
    fire_levels: List[int],                  # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    
    agent_y, agent_x = agent_pos
    num_tasks = len(fire_pos)

    best_task = 0
    best_task_score = -np.inf

    for task in range(num_tasks):
        fire_y, fire_x = fire_pos[task]
        
        # Calculate distance to fire
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        
        # Calculate potential suppression
        potential_suppression = min(agent_suppressant_num * agent_fire_reduction_power, fire_levels[task])

        # Calculate task severity score with a temperature parameter
        severity_temperature = 0.1
        severity = np.exp((fire_levels[task] - potential_suppression) / severity_temperature)
        
        # Calculate task weight score with a temperature parameter
        weight_temperature = 0.1
        weight_score = np.exp(fire_putout_weight[task] / weight_temperature)
        
        # Combine all factors into a final task score
        task_score = weight_score / (distance + 1e-9) - severity
        
        # If this task's score is the best so far, update best_task and best_task_score
        if task_score > best_task_score:
            best_task, best_task_score = task, task_score
            
    # Return the task with the highest score
    return best_task