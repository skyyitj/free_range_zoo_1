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
    
    # Initialize variables
    num_tasks = len(fire_pos)
    scores = []

    # Temperature parameters for score normalization
    distance_temp = 2.0
    fire_level_temp = 1.5
    reward_weight_temp = 1.0
    
    # Iterate through all fire tasks
    for i in range(num_tasks):
        # Step 1: Calculate Agent to Fire Distance
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = np.exp(-distance / distance_temp)  # Inverted distance score

        # Step 2: Fire Level Score (Criticality)
        fire_level_score = np.exp(fire_levels[i] / fire_level_temp)  # Give higher weight to high fire levels

        # Step 3: Reward Weight Score (Task Priority)
        reward_score = np.exp(fire_putout_weight[i] / reward_weight_temp)  # Reflect priority weights

        # Step 4: Combine Scores
        combined_score = distance_score * fire_level_score * reward_score
        
        # Append to scores list
        scores.append(combined_score)

    # Select task with the highest score
    selected_task = int(np.argmax(scores))
    
    return selected_task