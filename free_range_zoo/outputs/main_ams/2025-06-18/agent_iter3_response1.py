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

    # Hyperparameters for score normalization:
    distance_temp = 5.0  # Controls the influence of distance normalization
    intensity_temp = 2.0  # Controls the influence of fire intensity normalization
    priority_temp = 3.0  # Controls the influence of priority weight normalization

    # Calculate scores for each fire task
    scores = []
    for i in range(len(fire_pos)):
        # Compute agent distance to the fire
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))

        # Normalize distance using an exponential function
        normalized_distance = np.exp(-distance / distance_temp)

        # Normalize fire intensity using an exponential function
        normalized_intensity = np.exp(fire_intensities[i] / intensity_temp)

        # Normalize task priority weight using an exponential function
        normalized_priority = np.exp(fire_putout_weight[i] / priority_temp)

        # Calculate the effectiveness of the agent on the given fire
        remaining_suppressant = max(agent_suppressant_num, 0)  # Clamp suppressant to positive values
        fire_suppression_score = agent_fire_reduction_power * remaining_suppressant / (fire_intensities[i] + 1e-6)  # Avoid division by zero

        # Score formula: Combine normalized metrics with fire suppression score weighting
        score = (
            normalized_priority * fire_suppression_score +  # Prioritize based on weight
            normalized_distance * 0.5 +                     # Encourage closer fires
            normalized_intensity * 0.3                      # Factor intensity into decision
        )

        scores.append(score)

    # Select the fire task with the highest score
    best_task_index = int(np.argmax(scores))
    return best_task_index