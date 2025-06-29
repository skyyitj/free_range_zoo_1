def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current fire intensity level at each location
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
    num_tasks = len(fire_pos)
    
    # Define temperature parameters for score normalization
    distance_temperature = 1.0
    intensity_temperature = 5.0
    weight_temperature = 1.0

    # Calculate scores for each fire task
    task_scores = []
    for i in range(num_tasks):
        # Compute Euclidean distance from agent to fire location
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        
        # Compute adjusted fire intensity after applying suppressant
        adjusted_fire_intensity = max(0, fire_intensities[i] - agent_suppressant_num * agent_fire_reduction_power)
        
        # Normalize distance component (lower distance is better) and apply transformation
        distance_score = np.exp(-distance / distance_temperature)
        
        # Normalize fire intensity component (lower intensity is better) and apply transformation
        intensity_score = np.exp(-adjusted_fire_intensity / intensity_temperature)
        
        # Normalize priority weight (higher weight is better) and apply transformation
        weight_score = np.exp(fire_putout_weight[i] / weight_temperature)
        
        # Combine scores with weighted sum (distance, intensity, and priority weight)
        combined_score = distance_score + intensity_score + weight_score
        task_scores.append(combined_score)

    # Select the fire task with the highest score
    best_task_idx = np.argmax(task_scores)
    return best_task_idx