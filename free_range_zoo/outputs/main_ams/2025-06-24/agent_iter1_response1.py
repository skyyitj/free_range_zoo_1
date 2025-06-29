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
    """
    # Temperature parameters for score components
    intensity_temperature = 10.0
    distance_temperature = 0.5
    reward_temperature = 2.0

    # Current position of the agent
    agent_y, agent_x = agent_pos

    # Calculate scores for each fire task
    task_scores = []
    for i in range(len(fire_pos)):
        fire_y, fire_x = fire_pos[i]

        # Remaining fire intensity if this agent handles the fire
        remaining_fire = fire_intensities[i] - (agent_fire_reduction_power * agent_suppressant_num)

        # High penalty if fire self-extinguishes with excess damage
        if remaining_fire < 0:
            remaining_fire_penalty = abs(remaining_fire) * fire_putout_weight[i]
        else:
            remaining_fire_penalty = 0

        # Normalize fire intensity score using a temperature parameter
        normalized_intensity = fire_intensities[i] / intensity_temperature

        # Compute distance
        distance = ((fire_y - agent_y) ** 2 + (fire_x - agent_x) ** 2) ** 0.5
        normalized_distance = np.exp(-distance / distance_temperature)

        # Include reward weight
        normalized_reward = np.exp(fire_putout_weight[i] / reward_temperature)

        # Combine metrics into a single score
        score = normalized_reward * (normalized_intensity + normalized_distance) - remaining_fire_penalty

        task_scores.append(score)

    # Select the task with the highest score
    best_task_index = int(np.argmax(task_scores))
    return best_task_index