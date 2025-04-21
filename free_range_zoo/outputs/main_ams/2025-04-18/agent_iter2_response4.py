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
        # Calculate the Euclidean distance between the agent and the fire
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5

        # Fire risk factor considering the spread of the fire around the fire's position
        risk_factor = (1 + fire_intensity) ** 2
        # Suppression factor considering the agent's fire reduction power and available suppressants
        suppressant_factor = agent_fire_reduction_power * agent_suppressant_num / risk_factor

        # Final score considering the distance, fire risk, fire weight, and suppression factor
        score = fire_weight * suppressant_factor / (dist * risk_factor)
        
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire