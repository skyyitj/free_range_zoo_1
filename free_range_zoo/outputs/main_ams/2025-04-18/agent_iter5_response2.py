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

    temperature_dist = 0.05  # Lower temp_dist makes distance factor more prominent in score calculation. 
    temperature_fire_level = 2.0  # A higher fire level value leads to a more significant score difference between fire tasks. 

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor in task selection
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        # Include the fire level as an additional factor to prioritize fires with higher levels
        score = np.exp(fire_weight * suppression_power / (dist + 1))
        score *= np.exp(fire_level / temperature_fire_level)  # More weight is given to higher level fires
        score /= np.exp(dist / temperature_dist)  # Reducing the score of fires which are far away

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire