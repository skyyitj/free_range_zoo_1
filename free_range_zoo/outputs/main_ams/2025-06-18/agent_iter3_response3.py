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
    
    num_tasks = len(fire_pos)
    task_scores = []

    # === Temperature Parameters for Normalization ===
    intensity_temp = 1e-3
    weight_temp = 1e-2
    distance_temp = 1e-1

    # Iterate through all fire tasks and calculate scores
    for i in range(num_tasks):
        # Distance calculation
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)

        # Calculation of remaining fire intensity after suppression
        potential_intensity_reduction = (
            min(agent_suppressant_num, fire_intensities[i]) * agent_fire_reduction_power
        )
        remaining_intensity = max(0, fire_intensities[i] - potential_intensity_reduction)

        # Normalize score components
        normalized_distance = np.exp(-distance / distance_temp)
        normalized_weight = np.exp(fire_putout_weight[i] / weight_temp)
        normalized_intensity = np.exp(-remaining_intensity / intensity_temp)

        # Final score incorporates normalized components
        score = normalized_distance * normalized_weight * normalized_intensity
        task_scores.append(score)

    # Choose the task with the highest score
    best_task_index = np.argmax(task_scores)
    
    return best_task_index