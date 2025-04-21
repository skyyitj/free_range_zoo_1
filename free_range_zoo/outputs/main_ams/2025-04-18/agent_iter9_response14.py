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
    
    num_fires = len(fire_pos)  
    num_agents = len(other_agents_pos) + 1  # +1 for the current agent
    capacity_to_extinguish = agent_fire_reduction_power * agent_suppressant_num
    max_priority = -float('inf')
    selected_fire = None

    fire_level_temperature = 0.8 
    dist_temperature = 0.15  
    suppress_temperature = 0.05  

    for i in range(num_fires):
        fire_distance = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        suppression_capacity = min(fire_intensities[i], capacity_to_extinguish)
        fire_level = fire_levels[i]

        distance_score = np.exp(-dist_temperature * fire_distance) 
        suppression_score = np.exp(suppress_temperature * suppression_capacity)  
        level_score = np.exp(fire_level_temperature * fire_level)

        priority = fire_putout_weight[i] * (distance_score + suppression_score + level_score)

        if priority > max_priority:
            max_priority = priority
            selected_fire = i

    return selected_fire