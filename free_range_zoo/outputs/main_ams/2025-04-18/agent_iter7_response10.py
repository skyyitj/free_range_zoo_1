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

    # === Adjusting Scoring Criteria ===
    max_score = -float('inf')
    best_fire = None

    dist_temperature = 0.07  # Increase temperature coefficient to give more weight to the distance factor
    suppress_power_temperature = 0.08  # Decrease temperature to give less weight to the suppression power factor
    intensity_temperature = 0.2  # Introduce a new temperature coefficient for the fire intensity factor

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance factor with weight adjusted by distance temperature coefficient
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor with weight adjusted by suppression power temperature coefficient
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity+1)

        # Fire intensity factor with weight adjusted by intensity temperature coefficient
        intensity_factor = np.exp(fire_intensity / intensity_temperature)
        
        # Score calculation considering prioritization weight, firefighting efficiency, distance and intensity factors
        score = np.exp((fire_weight * suppression_power / (dist_temperature * dist + 1) + fire_level * intensity_factor) / suppress_power_temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire