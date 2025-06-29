def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity levels of each fire
    fire_intensities: List[float],               # Current intensity values of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    import numpy as np

    # Constants for transformation/normalization (customizable temperatures)
    distance_temp = 5.0
    intensity_temp = 10.0
    reward_temp = 5.0

    # Initialize the scores for all fire tasks
    scores = []

    for i in range(len(fire_pos)):
        # Extract fire task properties
        fire_y, fire_x = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        priority_weight = fire_putout_weight[i]

        # Step 1: Calculate the distance from the agent to the fire
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = np.exp(-distance / distance_temp)

        # Step 2: Factor in fire intensity and current level
        intensity_score = np.exp(fire_intensity / intensity_temp)

        # Step 3: Include priority weight as a measure of importance
        reward_score = np.exp(priority_weight / reward_temp)

        # Step 4: Calculate resource effectiveness (fire reduction and suppression)
        suppressant_effect = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensity)
        suppressant_score = suppressant_effect / fire_intensity if fire_intensity > 0 else 1.0

        # Step 5: Avoid high-intensity fires that may self-extinguish
        self_extinguish_penalty = 1.0 if fire_level < 10 else 0.1  # Arbitrary threshold for self-extinguish

        # Combine all scores into a weighted final score for the task
        score = (
            distance_score * 0.3 +  # Weight for proximity
            intensity_score * 0.3 +  # Weight for intensity
            reward_score * 0.2 +  # Weight for reward importance
            suppressant_score * 0.2  # Weight for suppressant effectiveness
        ) * self_extinguish_penalty

        scores.append(score)

    # Step 6: Select the task with the highest score
    best_task_index = np.argmax(scores)
    return best_task_index