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
    
    weight_for_saving_suppressant = 1.0  # Emphasizing on saving suppressant
    weight_for_preventing_burnout = 2.5  # Emphasize preventing fires from burning out naturally

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        required_suppressant = fire_intensity / agent_fire_reduction_power
        suppressant_efficiency = agent_fire_reduction_power / required_suppressant if required_suppressant else float('inf')
        
        suppressant_to_use = min(agent_supressant_num, required_suppressant)  # max suppressant the agent can use this turn
        
        suppression_potential = suppressant_to_use * suppressant_efficiency
        free_suppressant = agent_supressant_num - suppressant_to_use
        
        importance_weight = fire_putout_weight[task_index]
        
        task_score = (
            suppression_potential +
            free_suppressant * weight_for_saving_suppressant +
            importance_weight * 10.0 -
            distance * 0.1  # penalizing distance slightly
        )
        
        if fire_level > 5:  # If a fire is approaching critical levels
            task_score += fire_level * weight_for_preventing_burnout
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index