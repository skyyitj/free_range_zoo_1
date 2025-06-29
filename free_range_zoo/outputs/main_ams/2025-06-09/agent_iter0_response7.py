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
    import numpy as np

    num_tasks = len(fire_pos)
    
    # Define decision parameters
    max_intensity_temperature = 1.2
    max_suppression_power_temperature = 1.0
    fire_weight_temperature = 0.7
    distance_temperature = 0.5
    
    scores = []
    
    for i in range(num_tasks):
        # Calculate the distance from the agent to the fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        
        # Calculate normalized intensity score
        intensity_score = fire_intensities[i] / max(fire_intensities)
        intensity_score = np.exp(-intensity_score / max_intensity_temperature)
        
        # Calculate normalized fire level effect (higher levels might be critical)
        level_score = fire_levels[i] / max(fire_levels)
        level_score = np.exp(level_score / max_suppression_power_temperature)
        
        # Calculate weighted importance of putting out this fire
        weight_score = fire_putout_weight[i] / max(fire_putout_weight)
        weight_score = np.exp(weight_score / fire_weight_temperature)
        
        # Calculate penalty for distance
        distance_penalty = np.exp(-distance / max([np.sqrt((y - agent_pos[0])**2 + (x - agent_pos[1])**2) for y, x in fire_pos]) / distance_temperature)
        
        # Sum up the scores (here we can tweak the influence of each part)
        score = intensity_score * level_score * weight_score * distance_penalty
        scores.append(score)
    
    # Pick the fire with the highest score
    optimal_task_index = np.argmax(scores)
    return optimal_task_index