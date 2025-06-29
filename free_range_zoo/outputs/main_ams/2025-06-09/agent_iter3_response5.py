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

    # Adjust temperature for distance evaluation
    distance_temp = 3.0  # Adjusted to give less exponential significance to distance
    effectiveness_temp = 0.5  # Decreased to improve sensitivity as effectiveness is more crucial
    importance_temp = 0.2  # Decreased to give more importance to fire importance weight

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        # fire_level = fire_levels[task_index] # Unused variable
        fire_intensity = fire_intensities[task_index]
        
        # Calculate distance
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        
        potential_effectiveness = agent_fire_reduction_power * agent_suppressant_num
        
        # Calculate the "expected success" on the fire
        fire_improvement_ratio = min(potential_effectiveness / fire_intensity, 1.0)
        
        # Importance weight from the input
        importance_weight = fire_putout_weight[task_index]
        
        # Scoring adjustment to balance effectiveness, importance, and distance
        task_score = (
            -np.exp(distance/distance_temp) * 5.0 +
            np.exp(fire_improvement_ratio * importance_weight / effectiveness_temp) * 10.0 +
            np.exp(importance_weight / importance_temp)
        )
        
        # Replace existing task only if this score is higher - focusing on decisive changes
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index