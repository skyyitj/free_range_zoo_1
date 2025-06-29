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
    import numpy as np

    # Define temperature parameters for transforming score components
    distance_temperature = 10.0
    intensity_temperature = 0.5
    suppressant_temperature = 5.0
    priority_temperature = 1.0

    # Initialize a list to store scores for all tasks
    task_scores = []

    # Iterate over all fire locations
    for i in range(len(fire_pos)):
        # Extract fire position and relevant details
        fire_position = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        reward_weight = fire_putout_weight[i]

        # Calculate distance to fire
        distance = np.sqrt((fire_position[0] - agent_pos[0])**2 + (fire_position[1] - agent_pos[1])**2)
        distance_score = np.exp(-distance / distance_temperature)

        # Evaluate the intensity contribution (higher fire intensity means higher priority)
        intensity_score = np.exp(fire_intensity / intensity_temperature)

        # Check agent's ability to contribute to fire suppression based on available suppressant resources
        suppressant_contribution = agent_fire_reduction_power * agent_suppressant_num
        suppressant_score = np.exp(suppressant_contribution / suppressant_temperature)

        # Consider the reward weight of the fire task
        priority_score = np.exp(reward_weight / priority_temperature)

        # Combine scores into a single task score
        total_score = distance_score * intensity_score * suppressant_score * priority_score
        task_scores.append(total_score)

    # Select the task with the highest score
    selected_task_index = int(np.argmax(task_scores))
    return selected_task_index