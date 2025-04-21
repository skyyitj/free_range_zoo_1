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
        # Calculate euclidean distance from agent to each fire
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5
        
        # Add a factor to weight the fire_level in the score calculation
        # Higher fire_level means more urgency to put out this fire, therefore increasing its score
        level_factor = (10 - fire_level)/10

        # Score is now also influenced by the remaining fire suppressant
        # If an agent has less suppressants, it should prioritize closer fires, therefore we subtract the distance multiplied by the proportion of remaining suppressants
        suppressant_factor = agent_suppressant_num/10

        # Score = Weight × Suppression Potential - Distance × Fire Intensity + level_factor - suppressant_factor
        score = fire_weight * (agent_suppressant_num * agent_fire_reduction_power/fire_intensity) - suppressant_factor*dist*fire_intensity + level_factor

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire