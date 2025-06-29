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

    # === Normalization Temperature Parameters ===
    distance_temp = 10.0
    intensity_temp = 1.0
    weight_temp = 1.0

    # === Scoring System for Task Selection ===
    scores = []

    for i in range(len(fire_pos)):
        # Calculate the distance to the fire location
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)

        # Normalize distance using an exponential decay
        distance_factor = np.exp(-distance / distance_temp)

        # Normalize fire intensity
        intensity_factor = np.exp(fire_intensities[i] / intensity_temp)

        # Weight priority for this fire task
        weight_factor = np.exp(fire_putout_weight[i] / weight_temp)

        # Resource availability factor (if resource is close to exhaustion, prioritize larger fires)
        resource_factor = agent_suppressant_num / (fire_intensities[i] + 1)

        # Aggregate score for the task
        score = (
            weight_factor * intensity_factor * distance_factor * resource_factor
        )

        scores.append(score)

    # Select the index of the fire task with the highest score
    best_task_index = int(np.argmax(scores))
    return best_task_index