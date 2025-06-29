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
    import math

    num_tasks = len(fire_pos)  # Total fire locations
    scores = []

    # Temperature parameters for scoring components
    distance_temperature = 10.0  
    suppressant_temperature = 10.0  
    priority_temperature = 1.0  

    # Loop through all fire tasks to calculate scores
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]

        # 1. Distance to fire: Favor nearby fires
        agent_y, agent_x = agent_pos
        distance = math.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = math.exp(-distance / distance_temperature)

        # 2. Fire intensity and agent resources: Favor feasible tasks within suppressant capacity
        required_suppressant = fire_intensities[i] / agent_fire_reduction_power
        if required_suppressant > agent_suppressant_num:
            suppressant_score = 0  # Avoid infeasible tasks
        else:
            suppressant_score = math.exp(-required_suppressant / suppressant_temperature)

        # 3. Task priority: Use reward weight directly as an incentive
        priority_score = math.exp(fire_putout_weight[i] / priority_temperature)

        # Combine components into overall score (weighted sum)
        total_score = distance_score + suppressant_score + priority_score
        scores.append(total_score)

    # Select the task index with the highest score
    selected_task = max(range(num_tasks), key=lambda idx: scores[idx])

    return selected_task