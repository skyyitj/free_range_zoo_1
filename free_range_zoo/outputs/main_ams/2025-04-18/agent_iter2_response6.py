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
    # === Fire Task Selection Scoring ===
    max_score = float('-inf')
    best_fire = None
    
    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Adding the number of other agents targeting the same fire in the distance calculation
        num_other_agents = len([other_agent for other_agent in other_agents_pos if other_agent == fire_position])

        # Consider both the distance to the fire and the number of other agents targeting the same fire
        dist = (((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 
                + num_other_agents) / agent_suppressant_num

        # Add a factor of (agent_fire_reduction_power/fire_intensity) in the score calculation to improve 
        # Suppressant Efficiency and Average Fire Intensity Change
        suppressant_factor = (agent_fire_reduction_power / fire_intensity) * agent_suppressant_num
        score = fire_weight * suppressant_factor - dist * fire_intensity

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire