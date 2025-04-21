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

    max_score = -float('inf')
    best_fire = None

    # Reduced distance temperature
    dist_temperature = 0.1  

    # Increased suppress power_temperature
    suppress_power_temperature = 0.05  

    # Increased fire_level_temperature
    fire_level_temperature = 1.0  

    # Iterate over all fire positions
    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance factor with lower weight
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1e-7)
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity+1)

        # Score calculation
        score = np.exp((fire_weight * (fire_level+1e-7) / (dist_temperature * dist + 1) + suppression_power * suppress_power_temperature) / fire_level_temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire