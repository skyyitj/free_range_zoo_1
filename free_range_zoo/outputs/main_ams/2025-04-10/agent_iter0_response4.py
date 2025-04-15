def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    import math

    def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # Define upper fire level threshold (fires above this burn out naturally)
    MAX_ALLOWED_FIRE_LEVEL = 100.0

    best_task_index = -1
    best_score = float('-inf')

    for i, (f_pos, f_level, f_intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        # Skip if fire is too intense and will burn out on its own
        if f_level >= MAX_ALLOWED_FIRE_LEVEL:
            continue

        # Estimate how effective the agent would be on this task
        reduction_possible = min(agent_fire_reduction_power, f_level)
        if agent_suppressant_num <= 0 or reduction_possible <= 0:
            continue

        # Closer fires are prioritized
        dist = distance(agent_pos, f_pos)
        dist_score = -dist

        # Favor lower-intensity fires to improve success probability
        intensity_score = -f_intensity

        # Favor fires closer to burning out (but still salvageable)
        urgency_score = f_level

        # Compute total score with weights (tunable)
        score = (
            2.0 * urgency_score +
            1.5 * reduction_possible +
            1.0 * intensity_score +
            0.5 * dist_score
        )

        if score > best_score:
            best_score = score
            best_task_index = i

    # Fallback: if no good task found, pick the closest within range that’s not too dangerous
    if best_task_index == -1:
        min_dist = float('inf')
        for i, (f_pos, f_level) in enumerate(zip(fire_pos, fire_levels)):
            if f_level >= MAX_ALLOWED_FIRE_LEVEL:
                continue
            dist = distance(agent_pos, f_pos)
            if dist < min_dist:
                min_dist = dist
                best_task_index = i

    return best_task_index if best_task_index != -1 else 0