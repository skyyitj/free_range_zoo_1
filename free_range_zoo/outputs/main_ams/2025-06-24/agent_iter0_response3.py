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

    # Parameters for temperature-based normalization
    distance_temp = 10.0  # Temperature scaling for distance normalization
    intensity_temp = 5.0  # Temperature scaling for fire intensity normalization
    weight_temp = 3.0     # Temperature scaling for priority weight normalization

    # Determine the number of fire tasks
    num_tasks = len(fire_pos)

    # Store scores for each task
    task_scores = []

    # Iterate over all fire tasks and calculate a score
    for i in range(num_tasks):
        # Calculate distance of agent to the fire location
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = ((agent_y - fire_y) ** 2 + (agent_x - fire_x) ** 2) ** 0.5

        # Calculate the effective fire reduction at the task
        suppression_effectiveness = agent_fire_reduction_power * agent_suppressant_num
        remaining_fire_intensity = max(0, fire_intensities[i] - suppression_effectiveness)

        # Normalize score components using exponential scaling
        normalized_distance = -distance / distance_temp
        normalized_intensity = -remaining_fire_intensity / intensity_temp
        normalized_weight = fire_putout_weight[i] / weight_temp

        # Combine components into a single score
        score = (
            normalized_weight +  # High-priority task gets higher weight
            normalized_intensity +  # Fires with lower remaining intensity are preferred
            normalized_distance  # Closer fires are more favorable
        )

        task_scores.append(score)

    # Select the task with the highest score
    best_task_index = int(np.argmax(task_scores))

    return best_task_index