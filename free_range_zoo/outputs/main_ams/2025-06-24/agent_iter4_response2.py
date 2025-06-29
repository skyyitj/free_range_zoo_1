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
    
    # Define importance of features using temperature parameters
    distance_temperature = 5.0  # Distance importance scaling
    intensity_temperature = 2.0  # Fire intensity scaling
    reward_temperature = 1.0  # Reward weight scaling

    # Compute scores for each fire task
    num_tasks = len(fire_pos)
    scores = []
    for i in range(num_tasks):
        # Extract information about the fire task
        fire_coord = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_weight = fire_putout_weight[i]

        # Calculate the distance to the fire
        distance = np.linalg.norm([fire_coord[0] - agent_pos[0], fire_coord[1] - agent_pos[1]])
        # Normalize distance into a score (lower distance is better)
        distance_score = np.exp(-distance / distance_temperature)

        # Normalize fire intensity (higher intensity is more critical)
        intensity_score = np.exp(fire_intensity / intensity_temperature)

        # Incorporate priority weight directly
        reward_score = np.exp(fire_weight / reward_temperature)

        # Combine individual scores into an overall score
        # Prioritize based on intensity, closeness, and weight
        combined_score = (
            intensity_score * reward_score * distance_score
            if agent_suppressant_num > 0
            else 0  # If no suppressant left, deprioritize the task
        )

        scores.append(combined_score)

    # Choose the task with the highest score
    best_task_index = int(np.argmax(scores))
    return best_task_index