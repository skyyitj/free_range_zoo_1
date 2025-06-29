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

    num_tasks = len(fire_pos)
    scores = []

    # Temperature parameters for transforming score components
    intensity_temp = 1.0
    distance_temp = 5.0
    suppression_temp = 1.0

    for i in range(num_tasks):
        # Calculate distance between agent and fire location
        fire_dist = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)

        # Transform fire intensity for scoring
        scaled_fire_intensity = np.exp(fire_intensities[i] / intensity_temp)

        # Transform distance for scoring (inverse relationship: closer distance is better)
        distance_score = np.exp(-fire_dist / distance_temp)

        # Check if the agent can effectively suppress the fire at this location
        max_suppressible_fire = agent_fire_reduction_power * agent_suppressant_num
        suppressibility_score = (
            np.exp(min(max_suppressible_fire, fire_intensities[i]) / suppression_temp)
        )

        # Incorporate reward weight for prioritization
        reward_priority = fire_putout_weight[i]

        # Combine all factors into a single score
        total_score = (
            reward_priority * scaled_fire_intensity * distance_score * suppressibility_score
        )
        scores.append(total_score)

    # Choose the task with the highest score
    best_task_index = int(np.argmax(scores))
    return best_task_index