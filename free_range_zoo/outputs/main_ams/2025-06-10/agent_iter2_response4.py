from typing import Tuple, List
import numpy as np
import math

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float],
) -> int:
    num_tasks = len(fire_pos)

    best_task = -1
    best_score = float('-inf')
    
    # Constants (Temperature parameters for normalizing scores)
    distance_temperature = 1.0
    intensity_temperature = 1.0
    resource_temperature = 10.0
    urgency_temperature = 1.0

    for task_idx in range(num_tasks):
        yf, xf = fire_pos[task_idx]
        ya, xa = agent_pos
        
        # Distance from the agent to the fire task
        distance = math.sqrt((yf - ya) ** 2 + (xf - xa) ** 2)
        normalized_distance = np.exp(-distance / distance_temperature)
        
        # Fire intensity and level
        fire_intensity = fire_intensities[task_idx]
        fire_level = fire_levels[task_idx]
        normalized_intensity = np.exp(-fire_intensity / intensity_temperature)
        
        # Urgency and resources considerations
        can_extinguish_now = (agent_fire_reduction_power * agent_suppressant_num) > fire_intensity
        urgency = (1 if can_extinguish_now else 0)
        normalized_urgency = np.exp(urgency / urgency_temperature)
        
        # Reward weight for the task
        weight = fire_putout_weight[task_idx]
        
        # Effective score for each task
        score = (weight * normalized_distance * normalized_intensity * normalized_urgency)
        
        if score > best_score:
            best_score = score
            best_task = task_idx
    
    return best_task