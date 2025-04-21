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
    temperature = 0.1  # introduce a temperature parameter to adjust the soft-max function
    distance_weight = 2.0  # introduce a weight factor to scale distance impact in score

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor in task selection
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        # Score calculation after scaling distance by a weight factor
        # Include the fire level as an additional factor to prioritize fires with higher levels
        # Introduce a soft-max function with temperature to the score computation to amplify the differentiation between fires
        # Further modify the computation to account for the fire_putout_weight as the weight of distance factor and fire level
        # This allows for better balancing between focusing on high priority fires and efficient resource allocation.
        score = np.exp((fire_weight * suppression_power / (dist * distance_weight + 1) + fire_weight * fire_level) / temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire