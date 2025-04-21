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
  
    # Initialize maximum score and the best fire index
    max_score = float('-inf')
    best_fire = -1

    # Set the temperature parameter for the score calculation
    temperature = 0.05

    # Iterate through each fire
    for i in range(len(fire_pos)):
        # Calculate Euclidean distance from agent to the fire
        dist = ((fire_pos[i][0] - agent_pos[0])**2 + (fire_pos[i][1] - agent_pos[1])**2)**0.5
        
        # Assess the effectiveness score for this fire based on the agent's power, the fire's intensity and the agent's distance to the fire
        effective_score = fire_putout_weight[i] * (agent_fire_reduction_power / (fire_intensities[i] + 1)) * np.exp(-temperature * dist)
       
        # Take into account other agents' positions: if other agents are closer to this fire, the agent may choose another fire
        for other_agent_pos in other_agents_pos:
            other_agent_dist = ((fire_pos[i][0] - other_agent_pos[0])**2 + (fire_pos[i][1] - other_agent_pos[1])**2)**0.5
            if other_agent_dist < dist:
                effective_score *= 0.9

        # If there is not enough suppressant for this fire, penalize the score
        if fire_levels[i] > agent_suppressant_num:
            effective_score *= 0.5

        # Update the maximum score and best fire index
        if effective_score > max_score:
            max_score = effective_score
            best_fire = i

    return best_fire