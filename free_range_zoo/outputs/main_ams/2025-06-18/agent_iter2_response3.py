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
    # Temperature parameters for score weighting
    distance_temperature = 10.0
    intensity_temperature = 5.0
    priority_temperature = 3.0

    # Calculate scores for all fire tasks
    num_tasks = len(fire_pos)
    scores = []

    for i in range(num_tasks):
        # Distance component: Inverse of Manhattan distance to fire
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = abs(agent_y - fire_y) + abs(agent_x - fire_x)
        distance_score = np.exp(-distance / distance_temperature)
        
        # Fire intensity component: Higher intensity fires get higher priority
        fire_intensity_score = np.exp(fire_intensities[i] / intensity_temperature)
        
        # Reward weight component: Directly corresponds to task priority weights
        priority_weight_score = np.exp(fire_putout_weight[i] / priority_temperature)

        # Combine components into a single score
        total_score = distance_score * fire_intensity_score * priority_weight_score
        scores.append(total_score)

    # Select the task with the highest score
    selected_task_idx = int(np.argmax(scores))

    return selected_task_idx