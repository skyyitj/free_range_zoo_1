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
    fire_putout_weight: List[float],
) -> int:

    scores = []
    distance_temperature = 1.0  # Controls sensitivity to distance
    intensity_temperature = 0.5  # Controls sensitivity to fire intensity
    level_temperature = 1.0  # Controls sensitivity to fire level
    
    for index, (fire_position, fire_intensity, fire_level, weight) in enumerate(zip(fire_pos, fire_intensities, fire_levels, fire_putout_weight)):
        # Calculate Euclidean distance between the agent and the fire task
        distance = np.sqrt((agent_pos[0] - fire_position[0]) ** 2 + (agent_pos[1] - fire_position[1]) ** 2)
        # Normalize and scale distance
        distance_score = np.exp(-distance / distance_temperature)
        
        # Consider fire intensity
        intensity_score = np.exp(-fire_intensity / intensity_temperature)
        
        # Consider fire level
        level_score = np.exp(-fire_level / level_temperature)
        
        # Composite score calculation
        score = weight * distance_score * intensity_score * level_score
        scores.append(score)

    # Select the fire task with the highest score
    selected_task_index = np.argmax(scores)
    return selected_task_index