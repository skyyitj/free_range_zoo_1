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
    temperature_distance = 0.05    # Lower temperature parameter to further amplify the score differences
    temperature_suppression = 0.1  # Increased temperature value to downscale suppression power.

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor in task selection, the temperature value is increase for this metric to make it less dominate in the decision.
        suppression_power = np.exp(agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1))/ temperature_suppression

        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        score = fire_weight * suppression_power / (np.exp(dist + 1) / temperature_distance) + 2 * fire_level

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire