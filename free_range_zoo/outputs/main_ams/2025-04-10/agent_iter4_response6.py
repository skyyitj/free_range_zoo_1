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
    Returns the index of the fire task this agent should attempt to extinguish.
    """
    if not fire_pos or not fire_levels or not fire_intensities:
        return 0  # fallback: return first task if no data

    best_score = -float('inf')
    best_index = 0

    for i in range(len(fire_pos)):
        fire_y, fire_x = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]

        if fire_level <= 0:
            continue  # fire already extinguished

        # Skip fires that will extinguish naturally (high fire_level leads to penalty)
        if fire_level > 5.0:
            continue

        # Calculate distance
        dist = math.hypot(agent_pos[0] - fire_y, agent_pos[1] - fire_x)

        # Check if this agent has enough suppressant to help
        can_help = agent_suppressant_num >= 1 and agent_fire_reduction_power > 0

        # Score:
        # - prioritize manageable fires (not too high level)
        # - favor closer ones
        # - reward if fire is intense and agent can actually reduce it
        # - penalty if it's likely to be over-assigned
        agent_effectiveness = min(agent_fire_reduction_power, fire_level)
        teammate_distance_penalty = sum(
            1 for other in other_agents_pos if math.hypot(other[0] - fire_y, other[1] - fire_x) < 2.0
        )

        score = (
            5.0 * agent_effectiveness +
            3.0 * fire_intensity -
            1.5 * dist -
            2.0 * teammate_distance_penalty
        )

        # If agent can't help (no suppressant), apply penalty
        if not can_help:
            score -= 100.0

        if score > best_score:
            best_score = score
            best_index = i

    return best_index