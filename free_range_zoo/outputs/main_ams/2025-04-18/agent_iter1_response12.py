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
        
        # Score = Weight × Suppression Potential - Distance × Fire Intensity
        # Increased the contribution of agent's suppressant, aim to suppress the fire more positively
        suppression_potential = agent_suppressant_num * agent_fire_reduction_power/fire_intensity
        # Give higher weights to fires with more intensity, aim to prioritize more intense fires
        score = (0.7 * fire_weight * suppression_potential) - (0.3 * dist * fire_intensity)
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire