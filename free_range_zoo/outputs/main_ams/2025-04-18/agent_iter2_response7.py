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
        
        # Changes: Dist now also includes agent_fire_reduction_power as a factor and 
        # becomes a comprehensive indicator for distance, suppressant amount, and fire reduction power of the agent
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num*agent_fire_reduction_power)

        # Change: Redefine the suppressant_factor and how it is used in the score formula.
        # Now, it also considers fire's intensity, making it a composite multiplier for suppressant efficiency, fire intensity and agent's fire
        # reduction ability. This should decrease the "Average Fire Intensity Change" while increasing 
        # "Average Suppressant Efficiency".
        suppressant_factor = agent_suppressant_num * (agent_fire_reduction_power / fire_intensity)
        score = fire_weight * suppressant_factor - dist * fire_level

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire