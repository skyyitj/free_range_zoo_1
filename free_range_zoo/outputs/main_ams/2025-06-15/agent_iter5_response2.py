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
    import numpy as np
    
    # Parameters to control score normalization
    distance_temperature = 50.0
    intensity_temperature = 10.0
    reward_temperature = 1.0
    
    # Helper functions
    def euclidean_distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    num_tasks = len(fire_pos)
    scores = []
    
    for idx in range(num_tasks):
        # Distance-based component (agent's proximity to fire location)
        distance = euclidean_distance(agent_pos, fire_pos[idx])
        normalized_distance = np.exp(-distance / distance_temperature)
        
        # Fire intensity-based component (how difficult the fire is to suppress)
        normalized_intensity = np.exp(fire_intensities[idx] / intensity_temperature)
        
        # Reward weight component (task priority as given by environment)
        normalized_reward = np.exp(fire_putout_weight[idx] / reward_temperature)
        
        # Combine components into a single score
        # Higher proximity, higher reward weight, higher intensity => higher priority
        score = normalized_reward * normalized_intensity * normalized_distance
        
        # Append to scores list
        scores.append((score, idx))
    
    # Sort scores by priority and select the task with the highest score
    scores.sort(reverse=True, key=lambda x: x[0])
    best_task_idx = scores[0][1]
    
    return best_task_idx