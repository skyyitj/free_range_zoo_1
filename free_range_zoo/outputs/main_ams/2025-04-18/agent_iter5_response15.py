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

    # === Scoring Fire Tasks ===
    max_score = float('-inf')
    best_fire = None
    suppressant_temperature = 0.01  # Lower the temperature parameter to control suppressant usage
    distance_temperature = 0.05   # Lower the temperature parameter to prioritize closer fires
    level_temperature = 0.1  # Lower the temperature parameter to prioritize higher level fires

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = np.exp(-(fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2) / distance_temperature

        # Firefighting efficiency factor in task selection
        remaining_suppressant = np.exp(-(fire_intensity + 1) / (agent_fire_reduction_power * agent_suppressant_num)) / suppressant_temperature

        # Level of fire as a factor in task selection
        level_factor = np.exp(fire_level) / level_temperature

        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        score = fire_weight * dist * remaining_suppressant * level_factor

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire