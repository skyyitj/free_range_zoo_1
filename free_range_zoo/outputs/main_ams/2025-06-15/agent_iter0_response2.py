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

    Returns:
        int: The index of the selected fire task (0 to num_tasks-1)
    """
    import numpy as np

    num_tasks = len(fire_pos)
    task_scores = []

    # Parameters for transformations
    distance_temperature = 5  # Temperature for the distance exponential normalization
    suppressant_temperature = 2  # Temperature for remaining suppressant normalization
    fire_intensity_temperature = 1  # Temperature for fire intensity normalization

    for i in range(num_tasks):
        # Calculate distance to the fire location
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = np.exp(-distance / distance_temperature)

        # Compute max achievable fire reduction at this task location
        max_reduction = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensities[i])
        
        # Calculate the fire level after suppression
        remaining_fire_intensity = fire_intensities[i] - max_reduction
        suppressant_score = np.exp(-remaining_fire_intensity / suppressant_temperature)

        # Prioritize suppression of higher intensity fires
        fire_intensity_score = np.exp(fire_intensities[i] / fire_intensity_temperature)

        # Incorporate priority weights
        weight = fire_putout_weight[i]

        # Compute a combined score for the task
        combined_score = (
            weight * suppressant_score * fire_intensity_score * distance_score
        )
        task_scores.append(combined_score)

    # Choose the task with the highest score
    best_task_index = int(np.argmax(task_scores))
    return best_task_index