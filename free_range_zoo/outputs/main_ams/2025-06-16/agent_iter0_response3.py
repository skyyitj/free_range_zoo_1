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

    num_tasks = len(fire_pos)  # Total number of fire tasks
    best_task = 0              # Initialize the index of the selected fire task
    best_score = -np.inf       # Initialize the best score with a very small value

    # Temperature parameters for score transformation
    distance_temp = 0.5        # Controls sensitivity to distance
    intensity_temp = 0.8       # Controls sensitivity to fire intensity
    suppressant_temp = 1.2     # Controls sensitivity to suppressant
    reward_temp = 1.0          # Controls sensitivity to reward weight

    # Iterate over all fire tasks
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        distance = np.sqrt((fire_y - agent_pos[0])**2 + (fire_x - agent_pos[1])**2)

        # Compute the effective reduction in fire intensity if this agent is assigned
        effective_reduction = min(
            fire_intensities[i], agent_fire_reduction_power * agent_suppressant_num
        )
        
        # Fire remaining intensity after the agent's action
        remaining_intensity = fire_intensities[i] - effective_reduction

        # Score components:
        # 1. Distance penalty (inverse relationship)
        distance_factor = np.exp(-distance / distance_temp)

        # 2. Fire intensity factor (higher intensity = higher priority)
        intensity_factor = np.exp(fire_intensities[i] / intensity_temp)

        # 3. Resource allocation penalty (prioritize feasible tasks)
        if remaining_intensity > 0:
            suppressant_factor = np.exp(-remaining_intensity / suppressant_temp)
        else:
            suppressant_factor = 1.0  # Fully extinguished fires are ideal

        # 4. Reward weight factor (higher reward weights = higher priority)
        reward_factor = np.exp(fire_putout_weight[i] / reward_temp)

        # Compute the overall task score as a weighted combination of the above components
        score = distance_factor * intensity_factor * suppressant_factor * reward_factor

        # Update the best task if the current task has a higher score
        if score > best_score:
            best_score = score
            best_task = i

    return best_task