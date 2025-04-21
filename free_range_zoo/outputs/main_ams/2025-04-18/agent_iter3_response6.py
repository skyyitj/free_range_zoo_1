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
    
    # Initialise scoring
    max_score = float('-inf')
    best_fire = None
    
    # Iterate over each fire
    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        
        # Compute distance between agent and fire
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)
    
        # Score based on suppressant efficiency
        efficiency = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity+1)
        
        # Compute total score considering fire_weight
        score = fire_weight * efficiency / (dist + 1)

        # Update maximum score and choose corresponding fire
        if score > max_score:
            max_score = score
            best_fire = i
    
    return best_fire