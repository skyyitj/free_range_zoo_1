def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float], 
    fire_intensities: List[float]
) -> int:
    
    num_fires = len(fire_pos)

    if agent_suppressant_num <= 0:
        return -1
    
    task_scores = np.zeros(num_fires)

    for task_id in range(num_fires):
        fire_level = fire_levels[task_id]
        fire_intensity = fire_intensities[task_id]
        
        # Distance to the fire
        distance = np.sqrt((fire_pos[task_id][0]-agent_pos[0])**2 + (fire_pos[task_id][1]-agent_pos[1])**2)
        
        if fire_level > 5: # This fire will extinguish naturally, skip it
            continue

        suppressant_needed = fire_intensity / agent_fire_reduction_power
        if agent_suppressant_num < suppressant_needed: # This fire is too big for this agent, skip it
            continue
        
        # Score the fire: higher intensity is better, shorter distance is better
        task_scores[task_id] = fire_intensity / distance

    if np.all(task_scores == 0): # all fires are either too big or going to extinguish naturally
        return -1

    # Return the task with the highest score
    return np.argmax(task_scores)