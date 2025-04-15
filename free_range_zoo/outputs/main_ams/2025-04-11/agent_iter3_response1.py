def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float], 
    fire_intensities: List[float]
) -> int:
    # First, check if agent has suppressant left
    if agent_suppressant_num <= 0:
        return -1
    
    # Create a weighted distance score for each fire
    num_fires = len(fire_pos)
    fire_scores = []
    for i in range(num_fires):
        distance = np.sqrt((fire_pos[i][0] - agent_pos[0])**2 + (fire_pos[i][1] - agent_pos[1])**2)
        intensity = fire_intensities[i]
        # Weight the distance and fire intensity
        fire_score = (0.5 * distance) + (0.5 * intensity)
        fire_scores.append(fire_score)
    
    # Consider the positions of other agents
    for other_agent in other_agents_pos:
        distance = np.sqrt((other_agent[0] - agent_pos[0])**2 + (other_agent[1] - agent_pos[1])**2)
        if distance < agent_suppressant_num * agent_fire_reduction_power:
            # Prevent agents from choosing the same fire
            for i in range(num_fires):
                if np.sqrt((fire_pos[i][0] - other_agent[0])**2 + (fire_pos[i][1] - other_agent[1])**2) < distance:
                    fire_scores[i] += np.inf
    
    # Choose the fire with the minimum score
    return np.argmin(fire_scores)