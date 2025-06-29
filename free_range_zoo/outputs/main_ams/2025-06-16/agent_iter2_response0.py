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

    # Temperature parameters for transformations
    proximity_temperature = 10.0
    intensity_temperature = 5.0
    weight_temperature = 10.0

    num_tasks = len(fire_pos)
    scores = []

    for i in range(num_tasks):
        # Distance-based proximity score (lower distance → higher score)
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        proximity_score = np.exp(-distance / proximity_temperature)

        # Fire intensity score (higher intensity → higher score)
        intensity_score = np.exp(fire_intensities[i] / intensity_temperature)

        # Task priority weight score (higher weight → higher score)
        weight_score = np.exp(fire_putout_weight[i] / weight_temperature)

        # Combine scores with critical considerations
        # If agent's suppressant resources are insufficient for this task, deprioritize
        if agent_suppressant_num * agent_fire_reduction_power > fire_intensities[i]:
            resource_factor = 1.0  # Full support for this task
        else:
            resource_factor = 0.5  # Deprioritize due to insufficient resources

        # Final aggregated score
        combined_score = (proximity_score + intensity_score) * weight_score * resource_factor
        scores.append(combined_score)

    # Select task with the highest aggregated score
    selected_task_idx = int(np.argmax(scores))
    return selected_task_idx