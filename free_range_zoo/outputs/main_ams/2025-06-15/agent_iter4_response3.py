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
    import numpy as np

    # (1) Define temperature parameters for score normalization
    distance_temp = 10.0  # Higher values make distance less impactful
    intensity_temp = 1.0  # Lower values make fire intensity more granular
    weight_temp = 5.0     # Higher values emphasize reward weight

    num_tasks = len(fire_pos)
    best_task = None
    best_score = -np.inf

    for i in range(num_tasks):
        # (2) Calculate distance to the fire task
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = np.exp(-distance / distance_temp)

        # (3) Transform fire intensity using intensity and reduction power
        intensity_score = np.exp(-fire_intensities[i] / intensity_temp)

        # (4) Normalize reward weight
        weight_score = np.exp(fire_putout_weight[i] / weight_temp)

        # (5) Compute expected suppression effectiveness
        suppression_effectiveness = (agent_fire_reduction_power * agent_suppressant_num) / (fire_intensities[i] + 1)

        # (6) Combine all components into a single score
        total_score = (distance_score + weight_score + suppression_effectiveness) * intensity_score

        # (7) Update best task if this score is higher
        if total_score > best_score:
            best_score = total_score
            best_task = i

    return best_task