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
    
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]
    
    # Assign high initial value for minimum distance
    min_distance = np.inf
    min_distance_task = -1
    
    for task_id in range(num_fires):
        # Check if the fire is within reachable range
        if distances[task_id] < min_distance and fire_levels[task_id] > 0:
            min_distance = distances[task_id]
            min_distance_task = task_id
            
    # If no task is found within the agent's range
    if min_distance_task == -1:
        # Find the task with the highest fire level
        max_fire_level = max(fire_levels)
        min_distance_task = fire_levels.index(max_fire_level)
    
    return min_distance_task