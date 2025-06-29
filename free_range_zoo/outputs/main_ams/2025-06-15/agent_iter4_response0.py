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

    # Hyperparameters for transformations
    distance_temperature = 2.0    # Determines how distance impacts scoring
    intensity_temperature = 1.5   # Determines how fire intensity impacts scoring
    reward_temperature = 1.0      # Determines how reward weight impacts scoring

    # Initialize the best task index with -1 and the highest score with -infinity
    best_task_index = -1
    highest_score = float('-inf')

    # Iterate over all fire tasks
    num_tasks = len(fire_pos)
    for i in range(num_tasks):
        # Extract relevant fire task information
        fire_location = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        fire_weight = fire_putout_weight[i]

        # Compute the Euclidean distance from the agent to the fire location
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_location))

        # Normalize relevant parameters using transformations
        normalized_distance = np.exp(-distance / distance_temperature)  # Close fires get higher score
        normalized_intensity = np.exp(-fire_intensity / intensity_temperature)  # Lower intensity easier to extinguish
        normalized_reward = np.exp(fire_weight / reward_temperature)  # Higher reward weight preferred

        # Compute suppression effectiveness
        suppression_effectiveness = agent_fire_reduction_power * min(agent_suppressant_num, fire_level)

        # Combine components into a scoring metric
        task_score = (normalized_reward * suppression_effectiveness) * (normalized_distance + normalized_intensity)

        # Update best task selection if the current score is higher
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = i

    return best_task_index