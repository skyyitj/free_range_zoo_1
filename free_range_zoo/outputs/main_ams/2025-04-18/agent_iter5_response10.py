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

    max_score = float('-inf')
    best_fire = None

    # Lower the temperatures to better differentiate between different fires
    suppression_temp = 0.03
    weight_temp = 0.03

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance factor (incorporating suppressant into the distance factor to encourage the agent to move closer to the fire)
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor (using np.exp to amplify the differentiation)
        suppressant_needed = fire_intensity / agent_fire_reduction_power
        suppression_power = np.exp(agent_suppressant_num / suppressant_needed) if agent_suppressant_num < suppressant_needed else np.exp(suppressant_needed / agent_suppressant_num)
        suppression_power /= suppression_temp

        # Score calculation
        # Balance the weight and the firefighting efficiency (both using np.exp to amplify the differentiation)
        score = (np.exp(fire_weight) / weight_temp) * suppression_power - dist

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire