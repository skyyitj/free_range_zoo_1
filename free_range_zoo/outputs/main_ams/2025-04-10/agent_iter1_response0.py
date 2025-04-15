from typing import Tuple, List

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    # Return -1 if no fire exists
    if len(fire_pos) == 0:
        return -1

    # Agent cannot act if no suppressant
    if agent_suppressant_num <= 0:
        return -1

    # Parameters
    SELF_EXTINGUISH_LEVEL = 10.0
    RANGE_LIMIT = 5.0  # Only consider fires within 5 units
    MIN_EFFECTIVENESS_RATIO = 0.2  # Only attack if power is at least 20% of intensity

    # Build list of valid fire tasks
    valid_fires = []
    for i, (pos, level, intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        if level >= SELF_EXTINGUISH_LEVEL:
            continue  # Avoid fires that will extinguish naturally

        dy, dx = pos[0] - agent_pos[0], pos[1] - agent_pos[1]
        distance = (dy**2 + dx**2)**0.5

        if distance > RANGE_LIMIT:
            continue  # Skip distant fires

        effectiveness = agent_fire_reduction_power / (intensity + 1e-5)

        if effectiveness < MIN_EFFECTIVENESS_RATIO:
            continue  # Don't waste suppressant

        # Penalize targeting already crowded fires
        num_agents_nearby = sum(
            1 for a_pos in other_agents_pos
            if ((a_pos[0] - pos[0]) ** 2 + (a_pos[1] - pos[1]) ** 2) ** 0.5 < 2.0
        )

        # Priority: higher intensity, closer distance, fewer agents
        priority = (intensity / (distance + 1)) / (1 + num_agents_nearby)

        valid_fires.append((i, priority))

    if not valid_fires:
        return -1

    # Return the fire task with highest computed priority
    valid_fires.sort(key=lambda x: x[1], reverse=True)
    return valid_fires[0][0]