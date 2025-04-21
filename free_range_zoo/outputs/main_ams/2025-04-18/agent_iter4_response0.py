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
    temperature1 = 0.1  # introduce a temperature parameter to adjust the soft-max function for score
    temperature2 = 0.2  # introduce a second temperature parameter to adjust soft-max function for suppression power

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor in task selection
        suppression_power = np.exp(agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1) / temperature2)

        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        # Include the fire level as an additional factor to prioritize fires with higher levels
        # Introduce a soft-max function with temperature to the score computation to amplify the differentiation between fires
        # which have different scores
        score = np.exp((fire_weight * suppression_power / (dist + 1) + fire_level) / temperature1)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire