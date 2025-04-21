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
        
        # Change: added fire intensity in the calculation of dist
        # This would now consider the intensity of fire while calculating distance
        dist = (fire_intensity / ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5) / agent_suppressant_num

        # Change: Modified suppressant_factor in the score formula
        # Include fire_level in the suppressant_factor calculation to target higher intensity and uncontrolled fires
        suppressant_factor = (agent_fire_reduction_power / (fire_intensity+fire_level)) * agent_suppressant_num
        score = fire_weight * suppressant_factor - dist * fire_intensity
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire