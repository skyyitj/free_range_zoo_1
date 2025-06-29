def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float,
    agent_supressant_num: float, 
    other_agents_pos: List[Tuple[float, float]], 
    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    num_tasks = len(fire_pos)
    best_task_index = -1
    highest_score = float('-inf')
    
    # Define temperature constants easily adjustable for tuning
    distance_temp = 2.0
    effectiveness_temp = 1.8
    importance_temp = 3.0
    potential_burnout_temp = 2.5

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        importance_weight = fire_putout_weight[task_index]

        # Calculate the Euclidean distance to each fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Calculate potential effectiveness considering available suppressant
        potential_effectiveness = min(agent_fire_reduction_power * agent_supressant_num, fire_intensity)
        
        # Eagerly combat fires that are approaching burnout levels (i.e., high fire level)
        fire_level = fire_levels[task_index]
        potential_burnout_risk = np.exp(-fire_level / potential_burnout_temp)

        # Score calculation with emphasis on reducing burnout and attending high priority fires
        task_score = (
            (np.log(potential_effectiveness + 1) / effectiveness_temp) +
            (importance_weight * importance_temp) -
            (np.log(distance + 1) / distance_temp) +
            (potential_burnout_risk * 10)
        )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index