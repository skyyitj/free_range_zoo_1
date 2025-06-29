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
    
    # Temperature parameters for scaling each component in score calculation
    distance_temp = 10.0
    intensity_temp = 5.0
    weight_temp = 15.0

    # Compute normalized scores for all tasks
    task_scores = []
    for i in range(len(fire_pos)):
        fire_location = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_priority_weight = fire_putout_weight[i]

        # Calculate distance from agent's current position to fire location (Euclidean distance)
        distance = np.sqrt((fire_location[0] - agent_pos[0])**2 + (fire_location[1] - agent_pos[1])**2)

        # Normalize the distance score (lower distance is better)
        distance_score = np.exp(-distance / distance_temp)

        # Normalize fire intensity score (higher intensity is prioritized)
        intensity_score = np.exp(fire_intensity / intensity_temp)

        # Use task priority weight directly (higher weight is better)
        weight_score = np.exp(fire_priority_weight / weight_temp)

        # Combine scores into a weighted sum
        combined_score = (distance_score + intensity_score + weight_score) / 3.0
        task_scores.append(combined_score)

    # Select the task with the highest score
    best_task_index = np.argmax(task_scores)
    return best_task_index