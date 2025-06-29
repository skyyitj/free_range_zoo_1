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

    # Temperature parameters for score components
    priority_temp = 1.0
    proximity_temp = 10.0
    resource_efficiency_temp = 1.0

    def euclidean_distance(pos1, pos2):
        """Calculate Euclidean distance between two points (y1, x1) and (y2, x2)."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    # Total number of tasks
    num_tasks = len(fire_pos)
    
    # Scoring each task
    task_scores = []
    for i in range(num_tasks):
        # Priority score: Based on fire_putout_weight
        priority_score = np.exp(priority_temp * fire_putout_weight[i])
        
        # Proximity score: Closer distances are preferred
        distance = euclidean_distance(agent_pos, fire_pos[i])
        proximity_score = np.exp(-proximity_temp * distance)
        
        # Resource efficiency score: Based on expected suppressant usage
        required_suppressant = fire_intensities[i] / agent_fire_reduction_power
        if agent_suppressant_num >= required_suppressant:
            resource_efficiency_score = np.exp(resource_efficiency_temp * (-required_suppressant / agent_suppressant_num))
        else:
            resource_efficiency_score = 0  # Cannot efficiently handle fire if resources are insufficient
        
        # Combine the scores (with weights if needed)
        total_score = priority_score + proximity_score + resource_efficiency_score
        task_scores.append(total_score)

    # Choose the task with the highest score
    selected_task_index = np.argmax(task_scores)

    return selected_task_index