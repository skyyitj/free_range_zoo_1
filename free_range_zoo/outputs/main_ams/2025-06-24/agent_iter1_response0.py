def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y, x), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
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
    
    temperature_distance = 1.0    # Temperature parameter for distance-based score normalization
    temperature_intensity = 0.5   # Temperature parameter for fire intensity-based score normalization
    temperature_weight = 1.0      # Temperature parameter for task weight normalization
    
    num_tasks = len(fire_pos)
    scores = np.zeros(num_tasks)
    
    for i in range(num_tasks):
        # Distance score: closer fires are more important
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))
        distance_score = np.exp(-distance / temperature_distance)
        
        # Fire intensity score: prioritize fires with higher intensity
        intensity_score = np.exp(fire_intensities[i] / temperature_intensity)
        
        # Reward weight priority: prioritize based on given weights
        weight_score = np.exp(fire_putout_weight[i] / temperature_weight)
        
        # Remaining suppressant factor: balance with available resources
        suppressant_factor = min(agent_suppressant_num * agent_fire_reduction_power, fire_intensities[i])
        
        # Combine all components into a unified score
        scores[i] = distance_score * intensity_score * weight_score * suppressant_factor
    
    # Pick the fire task with the highest score
    chosen_task = np.argmax(scores)
    
    return chosen_task