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
    scores = []

    # Normalization temperature parameters
    distance_temp = 10.0       # Temperature for distance impact
    intensity_temp = 1.5       # Temperature for intensity consideration
    weight_temp = 2.0          # Temperature for reward weight prioritization

    for i in range(num_tasks):
        # Calculate distance score (inverted, closer fires are preferred)
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = np.exp(-distance / distance_temp)

        # Calculate fire intensity impact score (normalize higher intensities)
        fire_intensity = fire_intensities[i]
        intensity_score = np.exp(fire_intensity / intensity_temp)

        # Calculate reward weight score
        reward_weight = fire_putout_weight[i]
        weight_score = np.exp(reward_weight / weight_temp)
        
        # Estimate suitability based on available suppressant and fire reduction
        effective_reduction = agent_suppressant_num * agent_fire_reduction_power
        remaining_fire = fire_intensity - effective_reduction
        suppressant_impact_score = np.exp(-max(0, remaining_fire) / intensity_temp)

        # Combine scores: prioritize closer fires, higher weights, and manageable fires
        combined_score = distance_score * weight_score * suppressant_impact_score

        scores.append(combined_score)

    # Select task with the maximum score
    optimal_task_index = np.argmax(scores)

    return optimal_task_index