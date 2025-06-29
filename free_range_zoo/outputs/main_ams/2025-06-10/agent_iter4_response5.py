def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.

    Input Parameters:
        Agent Properties:
            agent_pos: (y, x) coordinates of the agent
            agent_fire_reduction_power: Fire suppression capability
            agent_suppressant_num: Available suppressant resources

        Team Information:
            other_agents_pos: List of (y, x) positions for all other agents
                            Shape: (num_agents-1, 2)

        Fire Information:
            fire_pos: List of (y, x) coordinates for all fires
                     Shape: (num_tasks, 2)
            fire_levels: Current fire intensity at each location
                        Shape: (num_tasks,)
            fire_intensities: Base difficulty of extinguishing each fire
                            Shape: (num_tasks,)

        Task Weights:
            fire_putout_weight: Priority weights for task selection
                               Shape: (num_tasks,)

    Returns:
        int: The index of the selected fire task (0 to num_tasks-1)
    """
    import numpy as np
    
    num_fires = len(fire_pos)
    scores = np.zeros(num_fires)

    dist_temperature = 0.1
    fire_level_temperature = 0.2
    suppression_effectiveness_temperature = 0.15
    
    for idx in range(num_fires):
        y_f, x_f = fire_pos[idx]
        y_a, x_a = agent_pos
        
        # Calculate distance and intended effect
        dist = np.sqrt((y_f - y_a)**2 + (x_f - x_a)**2)
        fire_combat_effect = agent_suppressant_num * agent_fire_reduction_power / (fire_intensities[idx] + 1)
        
        # Normalize values with exponentiated transformation to prioritize properly
        distance_score = np.exp(-dist * dist_temperature)
        fire_level_score = np.exp(fire_levels[idx] * fire_level_temperature)
        effectiveness_score = np.exp(fire_combat_effect * suppression_effectiveness_temperature)
        
        # Combine scores multiplying by the task weight to get a composite score
        scores[idx] = distance_score * fire_level_score * effectiveness_score * fire_putout_weight[idx]

    # Select the fire with the highest score
    chosen_fire_index = np.argmax(scores)
    
    return chosen_fire_index