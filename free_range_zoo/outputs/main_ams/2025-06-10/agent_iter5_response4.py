import numpy as np
from typing import Tuple, List

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
    
    if num_tasks == 0:
        return -1  # No tasks available

    # Arrays from the lists
    fire_positions = np.array(fire_pos)
    agent_position = np.array(agent_pos)
    
    # Calculate distances from agent to each fire
    distance = np.linalg.norm(fire_positions - agent_position, axis=1)

    # Calculate how effectively the agent can impact each fire
    max_possible_suppression = agent_fire_reduction_power * agent_suppressant_num
    effectiveness = np.clip(max_possible_suppression - np.array(fire_intensities), 0, None)
    
    # Normalization and transformation parameters
    temp_distance = 1.0
    temp_effectiveness = 3.0
    temp_suppressant = 2.0
    
    # Higher weight should result in higher score, inverse for distance
    normalized_distances = np.exp(-temp_distance * distance / distance.mean())
    normalized_effectiveness = np.exp(temp_effectiveness * effectiveness / effectiveness.mean() if effectiveness.mean() > 0 else effectiveness)
    normalized_weights = np.array(fire_putout_weight)
    
    # Combine components into a scoring system
    # Adding an emphasis on effectiveness and prioritized weights
    scores = (normalized_effectiveness * normalized_weights * normalized_distances)
    
    # Choose the fire with the highest score
    chosen_fire_index = np.argmax(scores)
    
    return chosen_fire_index