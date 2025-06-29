def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_supressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
):
    import numpy as np
    
    num_tasks = len(fire_pos)
    
    # Calculate distances between the agent and each fire location
    distances = np.array([
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ])
    
    # Normalize distances so closer fires are more attractive
    # Lower temperature to consider closer targets more heavily
    distance_temp = 2.5  # A more focused temperature scale
    distances_inv = np.exp(- distances / distance_temp)
    
    # Calculate suppression potential improvement per fire intensity
    suppression_effectiveness = agent_fire_reduction_power * agent_supressant_num / (fire_intensities + 1e-6)
    suppression_effectiveness = np.clip(suppression_effectiveness, 0, 1)  # Bound the effectiveness between [0,1]
    
    # Fire urgency is considered as inverse of fire level, the higher the level the more urgent
    urgency = 1 / (np.array(fire_levels) + 0.1)  # Add slight factor to avoid division by zero
    
    # The larger the fire weight, the more important the fire
    importance = np.array(fire_putout_weight)
    
    # Calculate complete scores for all tasks
    # Incorporating urgency and effectiveness more prominently
    scores = distances_inv * suppression_effectiveness * importance * urgency
    
    # Selecting the fire with the maximum score
    selected_task_index = np.argmax(scores)

    return selected_task_index