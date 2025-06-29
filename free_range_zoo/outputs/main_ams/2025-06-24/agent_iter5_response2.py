def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    import numpy as np

    # Temperature parameters for score normalization
    distance_temp = 0.2
    fire_intensity_temp = 0.5
    weight_temp = 1.0

    # Initialize the best score and best task index
    best_score = -np.inf
    best_task_idx = -1

    # Iterate over all fire tasks
    for i in range(len(fire_pos)):
        # Unpack fire information
        fire_location = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        priority_weight = fire_putout_weight[i]

        # Calculate Euclidean distance to the fire task
        distance = np.sqrt((agent_pos[0] - fire_location[0]) ** 2 + (agent_pos[1] - fire_location[1]) ** 2)
        # Transform and normalize the distance score (closer fires prioritized)
        normalized_distance_score = np.exp(-distance / distance_temp)

        # Transform and normalize fire intensity score (higher intensity prioritized)
        normalized_fire_intensity_score = np.exp(fire_intensity / fire_intensity_temp)

        # Transform and normalize reward weight (higher priority weight prioritized)
        normalized_weight_score = np.exp(priority_weight / weight_temp)

        # Suppression feasibility check
        suppression_capacity = agent_fire_reduction_power * agent_suppressant_num
        expected_remaining_fire = fire_intensity - suppression_capacity

        # Penalty if fire is expected to self-extinguish
        self_extinguish_penalty = 0
        if expected_remaining_fire > fire_level:  # Fire level exceeded, risk of self-extinguish
            self_extinguish_penalty = -1e3  # Assign a large penalty to dissuade selection

        # Combine scores into a weighted score
        score = (normalized_distance_score +
                 normalized_fire_intensity_score +
                 normalized_weight_score +
                 self_extinguish_penalty)

        # Update the best task index if this score is higher
        if score > best_score:
            best_score = score
            best_task_idx = i

    return best_task_idx