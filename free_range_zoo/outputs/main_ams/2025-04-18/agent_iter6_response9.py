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
    best_fire = -1
    temperature = 0.04  

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)
        
        # Add a factor of other agents' distance from the fire (to account for cooperation between agents)
        other_agents_dist = sum([((pos[0]-fire_position[0])**2 + (pos[1]-fire_position[1])**2)**0.5 / (len(other_agents_pos)+1) for pos in other_agents_pos])

        # Introduce a punish factor for high-intensity fires
        intensity_punish_factor = 1 / (1.5 ** fire_intensity)

        score = np.exp((fire_weight * suppression_power / (dist + 1 + other_agents_dist) + fire_level * intensity_punish_factor) / temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire