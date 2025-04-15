def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    # Go to recharge if suppressant is lower than 10% of the capacity
    if agent_suppressant_num < 0.1 * agent_fire_reduction_power:
        return -1 

    fire_scores = []
    for i, fire in enumerate(fire_pos):
        dist = np.sqrt((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2)
        # Score calculation based on intensity, distance and fire level
        score = fire_intensities[i] / (1 + dist) / (1 + fire_levels[i])
        fire_scores.append(score)

    chosen_fire = np.argmax(fire_scores)

    return chosen_fire