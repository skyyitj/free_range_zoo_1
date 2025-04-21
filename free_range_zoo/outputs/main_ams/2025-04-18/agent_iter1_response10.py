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
    best_score = float('-inf')
    best_fire = None

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # Calculate euclidean distance from agent to each fire
        dist = ((fire_position[0] - agent_pos[0])**2 + (fire_position[1] - agent_pos[1])**2)**0.5

        # Add penalty for fires that need more suppressant than agent has
        required_suppressant = fire_intensity / agent_fire_reduction_power
        suppressant_penalty = 0.0 if agent_suppressant_num >= required_suppressant else (required_suppressant - agent_suppressant_num)

        # Score is based on weight, reduction potential, distance, and suppressant availability
        score = fire_weight * (agent_suppressant_num * agent_fire_reduction_power / fire_intensity) - dist * fire_intensity - suppressant_penalty

        # Update best fire if current fire has higher score
        if score > best_score:
            best_score = score
            best_fire = i

    return best_fire