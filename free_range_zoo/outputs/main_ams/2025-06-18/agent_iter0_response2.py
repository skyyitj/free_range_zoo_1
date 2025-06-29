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
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    import numpy as np
    
    # Temperature parameters for scaling score components
    distance_temp = 1.0
    intensity_temp = 1.5
    weight_temp = 1.0
    suppressant_temp = 1.0

    # Helper function: Calculate Manhattan distance
    def manhattan_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Score for each fire task
    scores = []

    # Iterate over all fire tasks
    for i in range(len(fire_pos)):
        # Fetch fire properties
        fire_position = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        reward_weight = fire_putout_weight[i]

        # Step 1: Distance component
        distance = manhattan_distance(agent_pos, fire_position)
        distance_score = np.exp(-distance / distance_temp)  # Normalize closer fires as higher priority

        # Step 2: Intensity component
        intensity_score = np.exp(fire_intensity / intensity_temp)  # Higher intensity means higher priority

        # Step 3: Weight component
        weight_score = np.exp(reward_weight / weight_temp)  # Higher reward weight means higher priority

        # Step 4: Resource component
        suppress_power = agent_suppressant_num * agent_fire_reduction_power
        remaining_fire = fire_intensity - suppress_power
        suppressant_score = np.exp(-remaining_fire / suppressant_temp)  # Prefer tasks where the agent can suppress fire fully

        # Combine all components into a task score
        task_score = distance_score + intensity_score + weight_score + suppressant_score
        scores.append(task_score)

    # Select the fire task with the highest score
    best_task_index = int(np.argmax(scores))
    return best_task_index