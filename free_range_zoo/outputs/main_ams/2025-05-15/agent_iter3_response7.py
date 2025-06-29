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
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    # Temperature parameters
    distance_temp = 0.03  
    intensity_temp = 0.05
    level_temp = 0.07
    
    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        exp_distance = np.exp(-distance_temp * distance)
        
        fire_intensity = fire_intensities[i] + (fire_levels[i] / 10)
        exp_intensity = np.exp(-intensity_temp * fire_intensity)
        
        exp_level = np.exp(level_temp * fire_levels[i])

        # Calculate expected fire suppression score
        expected_suppression = agent_fire_reduction_power / (1.0 + fire_intensity)
        
        # Estimate number of agents targeting the same fire
        # The more agents potentially targeting the same fire, the less important it is for this agent to go there
        competing_agents = sum(
            np.sqrt((other_pos[0] - fire_pos[i][0])**2 + (other_pos[1] - fire_pos[i][1])**2) < distance for other_pos in other_agents_pos
        )
        non_competition_factor = 1 / (1 + competing_agents)
        
        # Here we adjust the score by resource availability
        resource_factor = agent_suppressant_num / (10 + fire_intensity)  # Adjusted term for suppressant levels

        # Combined score
        score = (fire_putout_weight[i] * expected_suppression * exp_distance * exp_intensity * exp_level * non_competition_factor * resource_factor)
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index