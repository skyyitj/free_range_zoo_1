def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    import numpy as np
    
    num_fires = len(fire_pos)
    
    # Define temperature parameters for normalization
    distance_temp = 10.0
    intensity_temp = 5.0
    reward_temp = 1.0
    
    # Initialize the list to store scores for each fire task
    scores = []
    
    # Iterate over all fire tasks
    for i in range(num_fires):
        # Distance between agent and fire
        fy, fx = fire_pos[i]
        ay, ax = agent_pos
        distance = np.sqrt((fy - ay)**2 + (fx - ax)**2)
        
        # Distance-based component (normalized)
        distance_score = np.exp(-distance / distance_temp)

        # Fire intensity component (normalized)
        fire_intensity_score = np.exp(-fire_intensities[i] / intensity_temp)
        
        # Reward weight component (normalized)
        reward_score = np.exp(fire_putout_weight[i] / reward_temp)
        
        # Combine scores weighted by their importance
        # Higher weights are given to priorities like reward and distance
        total_score = distance_score * 0.4 + fire_intensity_score * 0.2 + reward_score * 0.4
        
        # Append the calculated score for the current fire task
        scores.append(total_score)
    
    # Select the fire task with the highest score
    best_task = np.argmax(scores)
    
    return best_task