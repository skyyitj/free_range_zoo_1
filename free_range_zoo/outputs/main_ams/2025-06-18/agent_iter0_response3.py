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
    # Temperature parameters for scoring components
    intensity_temp = 1.0
    distance_temp = 1.0
    weight_temp = 1.0

    # Calculate scores for each fire task
    scores = []
    for i in range(len(fire_pos)):
        # Extract current fire information
        fire_intensity = fire_intensities[i]
        fire_priority_weight = fire_putout_weight[i]
        fire_loc = fire_pos[i]

        # Compute distance from agent to fire location
        agent_distance = ((agent_pos[0] - fire_loc[0]) ** 2 + (agent_pos[1] - fire_loc[1]) ** 2) ** 0.5

        # Score components
        intensity_score = fire_intensity / (fire_levels[i] + 1)
        distance_score = 1 / (agent_distance + 1)  # Safeguard against division by zero
        weight_score = fire_priority_weight

        # Transform scores using temperature parameters
        transformed_intensity = intensity_score ** (1 / intensity_temp)
        transformed_distance = distance_score ** (1 / distance_temp)
        transformed_weight = weight_score ** (1 / weight_temp)

        # Combine scores into a weighted sum
        task_score = transformed_intensity + transformed_distance + transformed_weight
        scores.append(task_score)

    # Select the fire task with the highest score
    best_task_index = int(scores.index(max(scores)))
    return best_task_index