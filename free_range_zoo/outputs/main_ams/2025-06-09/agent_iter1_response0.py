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
    best_task_index = -1
    highest_score = float('-inf')
    
    max_fire_level = max(fire_levels) if fire_levels else 0
    
    for task_index in range(num_tasks):
        fire_position = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        weight = fire_putout_weight[task_index]

        # Distance calculation
        distance = np.sqrt((agent_pos[0] - fire_position[0]) ** 2 + (agent_pos[1] - fire_position[1]) ** 2)
        
        # Effective reduction factor considering remaining suppressant and fire intensity
        possible_reduction = min(agent_suppressant_num, fire_intensity / agent_fire_reduction_power)
        
        # Adjusted weight factor considering actual suppression potential
        adjusted_weight = weight * (possible_reduction / fire_intensity) * fire_level / max_fire_level       
        
        # Numerically stable transformation of distances, weight, potential, and extra motivations
        distance_factor = np.exp(-distance / 20.0)  # makes agents prefer closer fires, normalized to grid scale
        weight_factor = np.exp(adjusted_weight * 1.5)  # boosted weight by adjusted weight and suppression potential
        suppression_potential_factor = np.exp((possible_reduction / fire_intensity) * 1.0) # Normalization of effect
        
        # Score composite
        task_score = distance_factor + weight_factor + suppression_potential_factor
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index
    
    return best_task_index