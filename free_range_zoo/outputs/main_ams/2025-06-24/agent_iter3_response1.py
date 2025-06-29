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
    
    # Temperature parameters for normalized scoring
    intensity_temp = 1.5
    distance_temp = 2.0
    reward_temp = 1.0
    
    # Scoring list
    task_scores = []
    
    # Iterate through all fire tasks
    for i in range(len(fire_pos)):
        # --- Fire Intensity Score ---
        # Higher intensity should be prioritized, normalized with exponential scaling
        fire_intensity_score = np.exp(fire_intensities[i] / intensity_temp)
        
        # --- Distance Score ---
        # Compute Euclidean distance to the fire, prioritize nearer fires
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        distance_score = np.exp(-distance / distance_temp)
        
        # --- Resource Availability ---
        # Check if agent can effectively fight the fire given remaining resources
        max_reduction = agent_fire_reduction_power * agent_suppressant_num
        effectiveness_score = 1.0 if max_reduction >= fire_intensities[i] else max_reduction / fire_intensities[i]
        
        # --- Reward Weight Score ---
        # Incorporate task reward weight (higher reward should increase score)
        reward_weight_score = np.exp(fire_putout_weight[i] / reward_temp)
        
        # --- Combined Score ---
        # Combine the scores with weighted importance
        combined_score = (
            fire_intensity_score * 0.5 +  # Fire intensity importance (50%)
            distance_score * 0.2 +        # Distance importance (20%)
            effectiveness_score * 0.2 +   # Allocation capability (20%)
            reward_weight_score * 0.1     # Reward weight importance (10%)
        )
        
        # Append the score to the list
        task_scores.append(combined_score)
    
    # Select the task with the highest score
    selected_task_index = int(np.argmax(task_scores))
    
    return selected_task_index