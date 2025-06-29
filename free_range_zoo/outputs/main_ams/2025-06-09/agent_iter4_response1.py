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

    # Adjust temperature tuning based on feedback
    distance_temp = 3.0  # Increasing sensitivity towards the distance impact
    effectiveness_temp = 0.5  # We need higher focus on suppressant effectiveness and utility
    importance_temp = 2.0  # Enhance the importance of task reward potential
    
    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate the Euclidean distance to each fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        
        # Use all suppressant if necessary for higher efficiency
        possible_suppressant_use = min(agent_supressant_num, fire_intensity / agent_fire_reduction_power)
        
        potential_effectiveness = agent_fire_reduction_power * possible_suppressant_use
        
        # Often, resource expenditure should be justified by a high potential reduction
        suppressant_utility = (potential_effectiveness - fire_intensity) / possible_suppressant_use if possible_suppressant_use > 0 else 0
        
        importance_weight = fire_putout_weight[task_index]

        # Adjust the score calculations
        task_score = (
            -np.log(distance + 1) / distance_temp +  # Emphasize proximity as a critical factor
            np.log(potential_effectiveness + 1) / effectiveness_temp +  # Normalize effectiveness with temperatures
            np.log(suppressant_utility + 10) * importance_temp +  # Boost score by suppressant utility.
            importance_weight * np.log(fire_level + 1) * 4.0  # Reward potential factored by fire severity
        )
        
        # Choose the task with the highest score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index