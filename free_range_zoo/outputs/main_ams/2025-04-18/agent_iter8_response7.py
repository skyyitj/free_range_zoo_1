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

    dist_temperature = 0.01  # Further reduce distance coefficient to prioritize closer fires
    suppress_power_temperature = 0.05  # Keep the same suppression power temperature to balance suppression power factor
    fire_level_temperature = 0.05  # Increase the fire levels temperature coefficient to prioritize high level fires

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Score calculation considering prioritization weight, firefighting efficiency, distance factor, and fire level.
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity+1)
        score = np.exp((fire_weight * suppression_power / (dist_temperature * dist + 1) + fire_level * fire_level_temperature) / suppress_power_temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire