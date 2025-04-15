from typing import List, Tuple
import math

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    """
    Determines the best action for an agent in the wildfire environment.
    """

    # If there are no fires, return -1 (or default to 0 to avoid crashes)
    if not fire_pos or not fire_levels or not fire_intensities:
        return 0

    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    best_score = float('-inf')
    best_index = 0  # fallback default

    for i, (f_pos, level, intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):

        # Skip already extinguished fires
        if level <= 0:
            continue

        # Skip fires that are too strong and will self-extinguish
        if level > intensity * 2:
            continue

        # Estimate how many other agents are close to this fire
        nearby_agents = sum(1 for a_pos in other_agents_pos if distance(a_pos, f_pos) < 2.0)

        # Avoid over-crowded fires
        if nearby_agents >= 2:
            continue

        # Compute potential effectiveness
        distance_penalty = distance(agent_pos, f_pos)
        if agent_suppressant_num <= 0:
            continue

        expected_reduction = agent_fire_reduction_power / intensity
        agent_score = expected_reduction - 0.2 * distance_penalty - 0.5 * nearby_agents

        # Prefer higher reward + manageable fires
        if agent_score > best_score:
            best_score = agent_score
            best_index = i

    return best_index