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
        
        # Calculate the average distance from this fire to other agents
        average_dist_other_agents = sum(((other_agent_pos[0]-fire_position[0])**2 + (other_agent_pos[1]-fire_position[1])**2)**0.5 for other_agent_pos in other_agents_pos) / len(other_agents_pos)

        # Score = Weight × Suppression Potential - Distance × Fire Intensity / average distance to other agents
        # Higher fire weight and suppression potential increase score, while higher distance and fire intensity decrease it
        # This time, we also penalize fires which are far from other agents, as they would be harder to deal with
        score = fire_weight * (agent_suppressant_num * agent_fire_reduction_power/fire_intensity) - (dist * fire_intensity/average_dist_other_agents)
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire