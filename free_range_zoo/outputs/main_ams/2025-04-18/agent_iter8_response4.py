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

    max_score = -float('inf')
    best_fire = None

    # Reducing distance factor weight and increasing fire level scaling factor
    dist_temperature = 0.08  # Temperature coefficient for balancing distance factor
    suppress_power_temperature = 0.04  # Temperature coefficient for balancing suppression power factor
    fire_level_temperature = 0.04  # Temperature coefficient for balancing fire level factor

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # distance factor
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # suppression power
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity+1)

        score = np.exp((fire_weight * suppression_power / (dist_temperature * dist + 1) + fire_level * fire_level_temperature) / suppress_power_temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire