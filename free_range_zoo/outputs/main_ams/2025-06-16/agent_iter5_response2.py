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

    # Parameters for score transformation
    distance_temperature = 1.0
    intensity_temperature = 1.5
    weight_temperature = 2.0

    # Compute scores for each fire location
    scores = []
    for i in range(len(fire_pos)):
        # Calculate Euclidean distance between agent and fire location
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        
        # Calculate potential fire reduction impact
        fire_reduction = min(agent_suppressant_num * agent_fire_reduction_power, fire_intensities[i])
        
        # Normalized distance score (lower distance is better)
        normalized_distance = np.exp(-distance / distance_temperature)
        
        # Normalized fire intensity score (higher intensity is better)
        normalized_intensity = np.exp(fire_intensities[i] / intensity_temperature)
        
        # Normalized reward weight (higher weight is better)
        normalized_weight = np.exp(fire_putout_weight[i] / weight_temperature)
        
        # Composite score combining priority factors
        score = normalized_distance * normalized_intensity * normalized_weight * fire_reduction
        scores.append(score)

    # Select the fire location with the highest score
    chosen_task_index = np.argmax(scores)

    return chosen_task_index