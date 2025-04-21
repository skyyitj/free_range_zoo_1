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

    best_score = -float('inf')
    best_task = None

    # temperature parameters for score components
    intensity_temperature = 0.5 
    distance_temperature = 1.0

    for i, ((fire_y, fire_x), fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # calculating distance between agent and fire
        dist = np.hypot(fire_y - agent_pos[0], fire_x - agent_pos[1])
        # calculating agent's relative strength to intensity of the fire
        relative_fire_strength = fire_level / (agent_fire_reduction_power * agent_suppressant_num)

        # score calculation
        score = fire_weight * np.exp(-intensity_temperature * relative_fire_strength) * np.exp(-distance_temperature * dist)
      
        if score > best_score:
            best_score = score
            best_task = i

    return best_task