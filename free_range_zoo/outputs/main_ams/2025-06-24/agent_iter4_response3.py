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

    # Initialize temperature parameters for score normalization
    distance_temperature = 5.0
    intensity_temperature = 2.0
    weight_temperature = 1.0

    # Define scoring for each fire task
    num_tasks = len(fire_pos)
    scores = []
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        priority_weight = fire_putout_weight[i]

        # Compute distance to fire
        agent_y, agent_x = agent_pos
        distance = np.sqrt((agent_y - fire_y)**2 + (agent_x - fire_x)**2)

        # Normalize distance score (lower distance should yield higher score)
        distance_score = np.exp(-distance / distance_temperature)

        # Normalize fire intensity score (higher intensity should yield higher score)
        intensity_score = np.exp(fire_intensity / intensity_temperature)

        # Normalize priority weight (higher weight should yield higher score)
        weight_score = np.exp(priority_weight / weight_temperature)

        # Combine scores into a total score
        total_score = (distance_score + intensity_score + weight_score) * fire_level
        scores.append(total_score)

    # Select the fire task with the highest score
    best_task_idx = np.argmax(scores)
    return best_task_idx