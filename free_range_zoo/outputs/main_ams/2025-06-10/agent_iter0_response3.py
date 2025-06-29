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
    """
    Optimal fire-fighting task selection for a single agent.
    """
    import numpy as np
    
    num_tasks = len(fire_pos)
    scores = np.zeros(num_tasks)
    
    # Calculation parameters
    distance_temperature = 1.0  # Controls sensitivity of softmax to distance
    intensity_temperature = 0.5  # Controls sensitivity of softmax to fire intensity
    level_temperature = 0.3      # Controls sensitivity of softmax to fire level
    
    for idx in range(num_tasks):
        # Calculate distance from agent to each fire
        distance = np.sqrt((agent_pos[0] - fire_pos[idx][0])**2 + (agent_pos[1] - fire_pos[idx][1])**2)
        
        # Calculate score components
        distance_score = np.exp(-distance / distance_temperature)
        intensity_score = np.exp(-fire_intensities[idx] / intensity_temperature)
        level_score = np.exp(-fire_levels[idx] / level_temperature)
        weight_score = fire_putout_weight[idx]

        # Combine scores by element-wise product
        scores[idx] = distance_score * intensity_score * level_score * weight_score
    
    # Select the fire task with the highest combined score
    selected_task_index = np.argmax(scores)
    
    return selected_task_index