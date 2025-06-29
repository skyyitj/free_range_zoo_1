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

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Distance calculation, we consider distance further for task selection
        distance = np.sqrt((agent_pos[0] - fire[0]) ** 2 + (agent_pos[1] - fire[1]) ** 2)
        
        # Strategic suppressant use, avoid over-suppressant expenditure where not needed
        possible_suppressant_use = min(
            agent_supressant_num, 
            min(fire_intensity, fire_level * agent_fire_reduction_power)
        )
        potential_effectiveness = possible_suppressant_use * agent_fire_reduction_power
        
        importance_weight = fire_putout_weight[task_index]

        # Adjust temperature values based on prior metrics observation:
        distance_temp = 1.0  # Lower distance importance as agents are fast-moving
        effectiveness_temp = 0.5  # Higher importance on being effective
        importance_temp = 0.5  # Weighted importance metric higher

        # Score calculations with emphasis on effectiveness and task priority adjustment
        task_score = (
            -distance / distance_temp +
            potential_effectiveness / effectiveness_temp +
            importance_weight * importance_temp
        )
        
        # Update the best task based on the highest score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index