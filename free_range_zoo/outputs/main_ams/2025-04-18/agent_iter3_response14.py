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

    # Initialize the best score and index
    best_score = -1
    best_index = -1

    for i in range(len(fire_pos)):
        # Calculate the Euclidean distance between the agent and the fire
        distance = ((agent_pos[0]-fire_pos[i][0])**2 + (agent_pos[1]-fire_pos[i][1])**2)**0.5

        # Calculate the potential impact of putting out the fire, considering the agent's power, the fire's intensity and the agent's remaining suppressants
        impact = (agent_fire_reduction_power * min(agent_suppressant_num, fire_levels[i])) / (fire_intensities[i] + 1)
       
        # Calculate the score for this fire, taking into account the fire's weight, the potential impact and the distance 
        score = fire_putout_weight[i] * impact / (distance + 1)

        # Update the best score and index if the current score is higher
        if score > best_score:
            best_score = score
            best_index = i

    return best_index