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

    # Total number of fires to help prioritize agents actions
    total_fires = len(fire_pos)

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # Calculate euclidean distance from agent to each fire
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5

        # Check if there are other agents closer to the fire
        other_agents_distance = [((pos[0]-fire_position[0])**2 + (pos[1]-fire_position[1])**2)**0.5 for pos in other_agents_pos]
        closest_other_agent_distance = min(other_agents_distance) if len(other_agents_distance) > 0 else dist

        # If there's another agent closer to the fire, decrease fire weight to discourage assignment
        if dist > closest_other_agent_distance:
            fire_weight *= 0.5

        # If agent is running low on suppressant, prioritize closer fires
        if agent_suppressant_num <= total_fires:
            dist = dist * 2

        # Score = Weight × Suppression Potential - Distance × Fire Intensity
        # Higher fire weight and suppression potential increase score, while higher distance and fire intensity decrease it
        score = fire_weight * (agent_suppressant_num * agent_fire_reduction_power/fire_intensity) - dist * fire_intensity
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire