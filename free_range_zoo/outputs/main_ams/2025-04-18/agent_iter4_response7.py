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
    distance_temperature = 0.2 # introduce a temperature parameter to adjust the distance-based factor
    weight_temperature = 0.1 # introduce a temperature parameter to adjust the priority-based component

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor in task selection
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        # Introduce temperature parameters to the distance factor and priority weight.
        # This will amplify the differentiation between tasks and steer the agents towards the preferable actions.
        # Use np.log1p function to smooth out the fire_level component. This can help prevent drastic changes in task selection
        score = np.exp((fire_weight / weight_temperature) * (suppression_power / (np.exp(dist / distance_temperature)) + np.log1p(fire_level)))

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire