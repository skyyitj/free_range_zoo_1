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

    # Define temperature parameters for normalized scoring
    distance_temperature = 10
    intensity_temperature = 5
    reward_temperature = 1

    # Determine best fire task based on scoring
    best_score = -np.inf
    best_task = -1

    # Iterate over all tasks
    for task_idx in range(len(fire_pos)):
        # Extract fire location and its properties
        fire_location = fire_pos[task_idx]
        fire_intensity = fire_intensities[task_idx]
        reward_weight = fire_putout_weight[task_idx]

        # Calculate distance between agent and fire
        distance = np.sqrt((agent_pos[0] - fire_location[0])**2 + (agent_pos[1] - fire_location[1])**2)
        
        # Transform distance (closer is better)
        distance_score = np.exp(-distance / distance_temperature)

        # Transform fire intensity (higher is more urgent)
        intensity_score = np.exp(fire_intensity / intensity_temperature)

        # Transform reward weight
        reward_score = np.exp(reward_weight / reward_temperature)

        # Combine all scores
        total_score = distance_score + intensity_score + reward_score

        # Consider available resources
        if agent_suppressant_num > 0:
            if total_score > best_score:
                best_score = total_score
                best_task = task_idx

    return best_task