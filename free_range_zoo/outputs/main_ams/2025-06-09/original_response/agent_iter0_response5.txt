from typing import Tuple, List
import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    num_tasks = len(fire_pos)
    
    # Settings for internal scores
    distance_temperature = 0.1  # Less sensitivity to distance.
    intensity_temperature = 0.5 # High sensitivity to fire intensity.
    
    best_task_index = -1
    max_score = -np.inf  # We want to maximize the score
    
    agent_y, agent_x = agent_pos
    
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        
        # Calculate the squared Euclidean distance (no need for sqrt, as we just do comparisons)
        distance = (fire_y - agent_y)**2 + (fire_x - agent_x)**2
        
        # Current fire intensity
        intensity = fire_intensities[i] * fire_levels[i]
        
        # Calculate weighted score for choosing this task
        # Adjust formulas based on hypothesized handy factors
        score = (
            fire_putout_weight[i] / np.exp(intensity_temperature * intensity) -
            np.exp(distance_temperature * distance)
        )
        
        # Choose task with the highest score
        if score > max_score:
            max_score = score
            best_task_index = i
    
    return best_task_index

# This policy function addresses balancing the nearness of the fire, fire intensity, and prioritizing by weight.
# Adjust the temperatures to modify the sensitivity to distance and intensity. High reward, close distance, and manageable intensity become the primary targets.