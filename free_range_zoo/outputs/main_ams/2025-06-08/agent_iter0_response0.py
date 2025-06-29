def single_agent_policy(
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

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

    # Scoring parameters and temperatures
    fire_intensity_temperature = 2.0  # For transforming fire intensity
    suppressant_temperature = 5.0    # For transforming available suppressant
    distance_temperature = 1.0       # For transforming distance scores

    # Position of the agent
    agent_y, agent_x = agent_pos

    # Calculate scores for each fire task
    num_tasks = len(fire_pos)
    task_scores = []
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        fire_weight = fire_putout_weight[i]
        
        # Distance score: inverse of distance between agent and fire location
        distance = np.sqrt((agent_y - fire_y)**2 + (agent_x - fire_x)**2)
        distance_score = np.exp(-distance / distance_temperature)  # Normalized using exp

        # Fire intensity score: higher intensity = higher priority
        intensity_score = np.exp(fire_intensity / fire_intensity_temperature)

        # Suppressant factor: prioritize fires that can be tackled with available resources
        max_reduction = agent_fire_reduction_power * agent_suppressant_num
        suppressant_score = np.exp(min(fire_intensity - max_reduction, 0) / suppressant_temperature)

        # Combine all factors: distance, intensity, suppressant usage, and reward weight
        combined_score = (
            fire_weight * intensity_score * suppressant_score * distance_score
        )
        task_scores.append(combined_score)

    # Select the fire task with the highest score
    best_task_index = int(np.argmax(task_scores))
    return best_task_index