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

    # Determine number of fires
    num_fires = len(fire_pos)

    # Initialize variables
    best_score = -float('inf')
    fire_to_handle = 0

    # For each fire, calculate a score considering the fire's intensity, distance to the fire and the amount of suppressant needed to handle it
    for i in range(num_fires):
        # Calculate the distance from the agent to the current fire
        distance = ((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2) ** 0.5

        # Calculate the amount of suppressant needed to handle the current fire
        suppressant_needed = fire_intensities[i] / agent_fire_reduction_power

        # Calculate the score for the current fire
        score = (fire_putout_weight[i] / distance) - suppressant_needed

        # If the calculated score is higher than the current best score, update the current best score and the fire to handle
        if score > best_score:
            best_score = score
            fire_to_handle = i

    # Return the fire to handle
    return fire_to_handle