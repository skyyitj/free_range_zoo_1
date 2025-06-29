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

    # Temperature parameters for scoring components
    dist_temp = 1.0
    suppression_temp = 1.0
    reward_temp = 1.0

    # Step 1: Compute a score for each fire task
    scores = []
    for i, fire_pos_i in enumerate(fire_pos):
        # Calculate the Euclidean distance between agent and fire
        dist = np.sqrt((agent_pos[0] - fire_pos_i[0])**2 + (agent_pos[1] - fire_pos_i[1])**2)

        # Calculate the potential suppression effect at this fire location
        potential_suppression = min(agent_suppressant_num * agent_fire_reduction_power, fire_intensities[i])

        # Calculate the remaining fire intensity after suppression
        remaining_fire = fire_intensities[i] - potential_suppression

        # Penalize tasks where remaining fire intensity is still high
        suppression_score = -np.exp(remaining_fire / suppression_temp)

        # Reward weights guide prioritization based on importance of task
        reward_score = np.exp(fire_putout_weight[i] / reward_temp)

        # Combine distance, suppression effect, and reward weight into a composite score
        # Shorter distance is better, so we use negative distance
        composite_score = reward_score + suppression_score - np.exp(dist / dist_temp)

        # Append the computed score for this task
        scores.append(composite_score)

    # Step 2: Select the task with the highest composite score
    chosen_task = int(np.argmax(scores))
    
    return chosen_task