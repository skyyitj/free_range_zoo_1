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
    
    num_tasks = len(fire_pos)  # Total number of fire tasks
    
    # Scoring function parameters
    distance_temp = 10  # Temperature for distance contribution
    intensity_temp = 2  # Temperature for intensity contribution
    weight_temp = 1.5   # Temperature for reward weight contribution

    # Initialize scores for each task
    task_scores = []
    
    for i in range(num_tasks):
        # Extract fire properties for the current task
        fire_position = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        reward_weight = fire_putout_weight[i]
        
        # Calculate distance to fire and normalize it
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_position))
        distance_score = np.exp(-distance / distance_temp)  # Closer fires get higher scores
        
        # Calculate intensity importance and normalize it
        intensity_score = np.exp(fire_intensity / intensity_temp)  # Higher intensity fires have higher scores
        
        # Incorporate reward weight
        weight_score = reward_weight ** weight_temp  # Increased influence from reward weight

        # Calculate expected effect of agent's suppressant resources
        suppress_score = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensity) / fire_intensity
        suppress_score = suppress_score ** 2  # Square to emphasize optimal suppression

        # Combine scores into a final priority score
        final_score = (distance_score * intensity_score * weight_score * suppress_score)
        
        # Append the score for this task
        task_scores.append(final_score)
    
    # Select the task with the highest score
    optimal_task_index = int(np.argmax(task_scores))
    
    return optimal_task_index