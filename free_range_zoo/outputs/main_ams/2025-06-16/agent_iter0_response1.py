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
    import numpy as np
    
    # Temperature parameters for score normalization
    distance_temp = 10.0
    intensity_temp = 0.5
    weight_temp = 1.0
    
    # Calculate scores for each fire task
    best_task_index = -1
    best_task_score = -float("inf")
    
    for i in range(len(fire_pos)):
        # Task information
        fire_position = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        reward_weight = fire_putout_weight[i]

        # Distance score (closer fires are prioritized)
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_position))
        distance_score = np.exp(-distance / distance_temp)

        # Intensity score (higher intensity fires are prioritized)
        intensity_score = np.exp(fire_intensity / intensity_temp)

        # Reward weight score (higher priority weights result in higher scores)
        weight_score = np.exp(reward_weight / weight_temp)
        
        # Feasibility: Can the agent contribute meaningfully to firefighting?
        effective_suppressant = min(agent_suppressant_num, fire_intensity / agent_fire_reduction_power)
        effectiveness_score = effective_suppressant * agent_fire_reduction_power

        # Aggregate score
        task_score = distance_score + intensity_score + weight_score + effectiveness_score

        # Select the best task based on the score
        if task_score > best_task_score:
            best_task_score = task_score
            best_task_index = i

    return best_task_index