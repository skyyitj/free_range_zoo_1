def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
) -> int:
    import math

    if agent_suppressant_num <= 0:
        # No suppressant available; avoid acting
        return -1  # No action

    NATURAL_EXTINGUISH_THRESHOLD = 10.0  # Fire level above this is assumed to go out naturally
    EFFECTIVE_RANGE = 5.0  # Max distance an agent can reach to apply suppressant

    def euclidean(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    best_score = float('-inf')
    chosen_task = -1

    for idx, (f_pos, f_level, f_intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        distance = euclidean(agent_pos, f_pos)

        if distance > EFFECTIVE_RANGE:
            continue  # Skip if fire is out of range

        if f_level >= NATURAL_EXTINGUISH_THRESHOLD:
            continue  # Avoid if likely to extinguish naturally

        # Check how many other agents are near this fire (to spread effort)
        crowding = sum(1 for other_pos in other_agents_pos if euclidean(other_pos, f_pos) < EFFECTIVE_RANGE)

        # Score: higher for closer, lower-intensity, low-crowd fires with manageable fire level
        score = (agent_fire_reduction_power / f_intensity) * max(0.1, (10 - f_level)) / (1 + crowding + distance)

        if score > best_score:
            best_score = score
            chosen_task = idx

    return chosen_task if chosen_task != -1 else 0  # Default to first task if no better found