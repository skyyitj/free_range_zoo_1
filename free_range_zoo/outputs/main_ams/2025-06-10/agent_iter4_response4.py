def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
):
    import numpy as np
    
    num_tasks = len(fire_pos)
    
    scores = np.zeros(num_tasks)
    
    # Temperate parameters to adjust the sensitivity in calculations
    distance_temp = 0.1
    intensity_temp = 1.0
    difficulty_temp = 0.5
    resource_temp = 1.0
    weight_temp = 1.0
    
    # Calculate scores for each fire task
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        
        # Calculate Euclidean distance from the agent to the fire
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        
        current_level = fire_levels[i]
        current_intensity = fire_intensities[i]
        
        # Factor in the fire putout weight
        task_priority_weight = fire_putout_weight[i]
        
        # Compute task score components to minimize distance and level and maximize priority weight
        distance_score = np.exp(-distance_temp * distance)
        level_score = np.exp(-intensity_temp * current_level)
        difficulty_score = np.exp(-difficulty_temp * current_intensity)
        
        # Resource weighting: more suppressant available => higher confidence that this agent effectively fight the fire
        resource_weight = agent_suppressant_num if agent_suppressant_num > 0 else 0.1
        resource_score = np.exp(resource_temp * resource_weight)
        
        priority_and_resource_score = task_priority_weight * resource_score
        
        # Combine components
        # Note: component weights (alpha, beta, gamma ...) could be fine-tuned or learned.
        scores[i] = (distance_score * 0.25 + level_score * 0.25 + priority_and_resource_score * 0.50) * difficulty_score
    
    # Pick the task with the highest score
    top_task_index = np.argmax(scores)
    
    return top_task_index