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

    dist_scale = 1.0  # scaling factor for distance
    weight_scale = 3.0  # scaling factor for task weight
    power_scale = 1.0  # scaling factor for suppression power
    level_scale = 0.5  # scaling factor for fire level

    # === Scoring Fire Tasks ===
    max_score = float('-inf')
    best_fire = None

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor in task selection
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        # Score calculation taking into consideration weights, firefighting efficiency, and distance factor
        # Included fire level as an additional factor with a scaling factor
        # The scaling factors are used to adjust importance of each factor in score computation
        score = weight_scale * fire_weight + power_scale * suppression_power / (dist_scale * dist + 1) + level_scale * fire_level

        # If the computed score is higher than the current max score, the score is now the max_score and i is the best_fire
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire