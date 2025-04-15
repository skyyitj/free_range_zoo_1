def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],

    # Task information
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    """
    Determines the best action for an agent in the wildfire environment.
    """
    import math

    if agent_suppressant_num <= 0:
        # No suppressant available, cannot act
        return -1  # No action

    def euclidean(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    best_score = float('-inf')
    best_index = 0

    for i, (f_pos, f_level, f_intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        if f_level >= 10:
            # Too high, likely to extinguish naturally — skip to avoid penalty
            continue

        # Count how many agents are likely targeting or near this fire
        nearby_agents = sum(1 for pos in other_agents_pos if euclidean(pos, f_pos) < 3.0)

        # Calculate effectiveness score
        distance = euclidean(agent_pos, f_pos)
        if distance > 5.0:
            continue  # Fire is out of effective range

        suppression_ratio = agent_fire_reduction_power / f_intensity
        urgency = f_level * f_intensity
        collaboration_penalty = 0.5 ** nearby_agents  # Reduce reward if many agents are nearby

        score = suppression_ratio * urgency * collaboration_penalty / (distance + 1e-3)

        if score > best_score:
            best_score = score
            best_index = i

    return best_index