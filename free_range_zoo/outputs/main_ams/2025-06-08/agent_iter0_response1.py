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
    
    num_tasks = len(fire_pos)
    
    # Parameters for transformation functions
    distance_temperature = 5.0   # Controls scaling of distance impact on score
    intensity_temperature = 10.0  # Controls scaling of intensity impact on score
    suppressant_temperature = 3.0  # Controls scaling of suppressant availability impact
    
    # Initialize variables to store scores
    scores = []
    
    for i in range(num_tasks):
        # Calculate distance from agent to fire location
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y) ** 2 + (fire_x - agent_x) ** 2)

        # Transform distance score (closer fires preferred)
        transformed_distance = np.exp(-distance / distance_temperature)
        
        # Transform fire intensity score (higher intensity fires are more urgent)
        transformed_intensity = np.exp(fire_intensities[i] / intensity_temperature)
        
        # Check remaining suppressant capacity impact (focus on fires that can be fully extinguished)
        expected_suppressant_use = min(agent_suppressant_num, fire_intensities[i] / agent_fire_reduction_power)
        suppressant_impact = np.exp(expected_suppressant_use / suppressant_temperature)
        
        # Combine scores with priority weights
        score = fire_putout_weight[i] * transformed_distance * transformed_intensity * suppressant_impact
        
        scores.append(score)
    
    # Select task index corresponding to maximum score
    best_task_index = int(np.argmax(scores))
    
    return best_task_index