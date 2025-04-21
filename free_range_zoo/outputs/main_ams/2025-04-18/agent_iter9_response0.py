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

    dist_temperature = 0.3  # Adjusted to give more importance to distance.
    suppress_power_temperature = 0.05  # Adjusted to balance out suppressing power's effect.
    fire_level_temperature = 1.0  # Adjusted to maintain high importance of fire level.
    intensity_temperature = 0.2  # New component to factor in fire intensities.

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1e-7)
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity+1)
        fire_intensity_factor = fire_intensity / (dist_temperature * dist + 1)  # New scoring component for fire intensity.

        score = np.exp((fire_weight * (fire_level+1e-7) / (dist_temperature * dist + 1) + fire_intensity_factor * intensity_temperature + suppression_power * suppress_power_temperature) / fire_level_temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire