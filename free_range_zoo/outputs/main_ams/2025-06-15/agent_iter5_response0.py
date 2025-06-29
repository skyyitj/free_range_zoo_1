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

    Returns:
        int: The index of the selected fire task (0 to num_tasks-1)
    """
    # Scoring parameters
    distance_temperature = 10.0  # Temperature for distance normalization
    intensity_temperature = 5.0  # Temperature for intensity normalization
    priority_temperature = 3.0   # Temperature for priority weight normalization

    num_tasks = len(fire_pos)
    scores = []

    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos

        # Calculate distance between agent and fire location
        distance = ((agent_y - fire_y)**2 + (agent_x - fire_x)**2)**0.5
        distance_score = -np.exp(-distance / distance_temperature)  # Penalize farther fires

        # Normalize fire intensity
        intensity_score = np.exp(fire_intensities[i] / intensity_temperature)

        # Priority weight normalization
        priority_score = np.exp(fire_putout_weight[i] / priority_temperature)

        # Combine scores with future suppression estimate
        estimated_fire_remaining = max(fire_intensities[i] - agent_fire_reduction_power * agent_suppressant_num, 0)
        agent_effectiveness_score = -np.exp(estimated_fire_remaining / intensity_temperature)  # Favor tasks likely to be extinguished completely

        # Total score
        total_score = priority_score + intensity_score + distance_score + agent_effectiveness_score
        scores.append(total_score)

    # Select the task with the highest score
    return int(np.argmax(scores))