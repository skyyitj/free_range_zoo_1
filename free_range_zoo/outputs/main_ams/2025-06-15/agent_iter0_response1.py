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

    # Helper function to calculate Euclidean distance
    def distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    # Initialize task scores
    task_scores = []

    # Temperature parameters for score components
    distance_temp = 1.0
    intensity_temp = 1.0
    weight_temp = 1.0

    # Iterate through all fire tasks
    for i, fire in enumerate(fire_pos):
        # Calculate distance between agent and fire location
        dist = distance(agent_pos, fire)
        dist_score = np.exp(-dist / distance_temp)  # Transform distance (closer fires preferred)

        # Calculate intensity score (prioritize higher intensity fires)
        intensity_score = np.exp(fire_intensities[i] / intensity_temp)

        # Calculate reward weight score (higher weights preferred)
        weight_score = np.exp(fire_putout_weight[i] / weight_temp)

        # Combine scores with remaining suppressant considerations
        if fire_levels[i] > 0 and agent_suppressant_num > 0:
            # Estimate fire reduction potential
            reduced_fire = fire_intensities[i] - (agent_fire_reduction_power * agent_suppressant_num)
            fire_reduction_score = max(0, reduced_fire)  # Penalize if fire can't be fully suppressed
        else:
            fire_reduction_score = float('inf')  # Avoid assigning tasks if no suppressant is left

        # Combine scores into a single metric
        combined_score = (weight_score * intensity_score * dist_score) / (1 + fire_reduction_score)
        task_scores.append(combined_score)

    # Select the task with the maximum score
    selected_task_index = int(np.argmax(task_scores))
    return selected_task_index