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
    temperature_a = 0.05  # Adjust the soft-max function scales for different components 
    temperature_b = 0.15 

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection, add a factor of square root of the fire level to the distance, considering fires with higher
        # level are more likely to spread
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2 + fire_level**0.5) / (agent_suppressant_num+1)

        # Firefighting efficiency factor in task selection
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        # Adjust the soft-max function with different temperatures according to fire_level and dist factors to address the
        # imbalance between them
        score = np.exp((fire_weight * suppression_power / np.exp(dist / temperature_b)) + fire_level / temperature_a)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire