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

    # Define temperature parameters for score normalization
    intensity_temperature = 10.0    # Temperature parameter for fire intensity scaling
    distance_temperature = 5.0      # Temperature parameter for distance scaling
    priority_temperature = 2.5      # Temperature parameter for priority scaling

    # Initialize variables
    num_tasks = len(fire_pos)
    scores = []  # Stores the score for each task

    # Iterate over each fire task
    for i in range(num_tasks):
        # Compute distance to the fire location (Euclidean distance)
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)

        # Compute remaining fire intensity after agent's suppression attempt
        remaining_intensity = max(
            0, fire_intensities[i] - agent_suppressant_num * agent_fire_reduction_power
        )

        # Assign task priority using weight and remaining intensity
        intensity_score = np.exp(-remaining_intensity / intensity_temperature)
        distance_score = np.exp(-distance / distance_temperature)
        priority_score = np.exp(fire_putout_weight[i] / priority_temperature)

        # Combine scores into a weighted task score
        task_score = intensity_score * priority_score * distance_score
        scores.append(task_score)

    # Select the task with the highest score
    best_task_index = int(np.argmax(scores))
    return best_task_index