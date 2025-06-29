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

    # Initialize scores for each fire
    scores = []

    # Hyperparameters (temperature for score adjustment)
    distance_temp = 1.0
    intensity_temp = 1.0
    weight_temp = 1.0

    # Loop over all fire tasks to calculate a score for each
    for i in range(len(fire_pos)):
        # Calculate distance between agent and fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        distance_score = np.exp(-distance / distance_temp)  # Closer fires get higher scores

        # Normalize fire intensity based on fire reduction power and suppressant available
        suppressable_intensity = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensities[i])
        intensity_ratio = suppressable_intensity / fire_intensities[i]
        intensity_score = intensity_ratio ** intensity_temp  # Fires that can be controlled better get higher scores

        # Use fire priority weight to scale the score
        weight_score = fire_putout_weight[i] ** weight_temp

        # Combine scores for a unified priority score
        combined_score = distance_score * intensity_score * weight_score
        scores.append(combined_score)

    # Select the fire task with the highest score
    best_task_idx = int(np.argmax(scores))
    return best_task_idx