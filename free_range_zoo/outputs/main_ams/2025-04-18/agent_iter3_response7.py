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

    # === Scoring Fire Tasks ===
    max_score = float('-inf')
    best_fire = None

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num + 1)
        
        # Fire intensity and level-based factor for task selection
        fire_power = fire_intensity * (fire_level / (agent_fire_reduction_power * agent_suppressant_num + 1))
        
        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        score = fire_weight * fire_power / (dist + 1)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire