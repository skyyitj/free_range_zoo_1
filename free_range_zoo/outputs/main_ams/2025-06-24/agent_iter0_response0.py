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
    # Define temperature variables for transforming score components
    intensity_temp = 1.0
    distance_temp = 1.0
    priority_temp = 1.0

    # Define the euclidean distance function
    def euclidean_distance(agent_pos, fire_pos):
        return ((agent_pos[0] - fire_pos[0])**2 + (agent_pos[1] - fire_pos[1])**2)**0.5

    # Calculate scores for each fire task
    scores = []
    for i in range(len(fire_pos)):
        # Compute the reduction value if the agent chooses this fire task
        effectiveness = agent_fire_reduction_power * min(agent_suppressant_num, fire_intensities[i])

        # Estimate remaining fire intensity
        remaining_fire_intensity = max(0, fire_intensities[i] - effectiveness)

        # Compute proximity score (minimize distance to the fire)
        distance = euclidean_distance(agent_pos, fire_pos[i])
        distance_score = np.exp(-distance / distance_temp)

        # Compute fire intensity score (prioritize higher intensity tasks)
        intensity_score = np.exp(fire_intensities[i] / intensity_temp)

        # Compute priority score based on fire_putout_weight
        priority_score = np.exp(fire_putout_weight[i] / priority_temp)

        # Combine scores (higher scores = better tasks)
        task_score = priority_score + intensity_score + distance_score - remaining_fire_intensity
        scores.append(task_score)

    # Choose the task with the maximum score
    chosen_task = int(np.argmax(scores))

    return chosen_task