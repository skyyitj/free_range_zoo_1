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

    max_score = -1e9
    best_fire = -1
    
    temperature_fire = 0.05 
    temperature_agent = 0.0005 
    temperature_dist = 0.3 

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        distance_score = np.exp(-dist / temperature_dist) 

        fire_score = np.exp((fire_weight * suppression_power) / (temperature_agent+1))
        
        intensity_score = np.exp(fire_level / (temperature_fire + 1)) 
        
        score = distance_score + fire_score + intensity_score

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire