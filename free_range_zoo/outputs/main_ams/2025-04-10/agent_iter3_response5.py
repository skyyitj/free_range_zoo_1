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
    Improved fire-fighting policy with collaboration and resource management.
    """
    if agent_suppressant_num <= 0:
        return -1  # No suppressant left, do nothing

    def distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    best_index = -1
    best_score = -float("inf")

    for i, (fpos, flevel, fintensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        dist = distance(agent_pos, fpos)
        if dist > 5:  # Assume effective suppressant range is 5 units
            continue

        # Avoid fires too intense that will go out naturally (threshold is arbitrary here)
        if flevel >= fintensity * 1.5:
            continue  # Skip to avoid penalty

        # Count other agents near this fire
        nearby_agents = sum(1 for apos in other_agents_pos if distance(apos, fpos) < 5)

        # Score favors closer, lower-intensity fires with less competition
        score = (
            -dist  # prefer close fires
            - fintensity * 2  # harder fires are deprioritized
            - nearby_agents * 1.5  # avoid overlap
            + max(0, (agent_fire_reduction_power - fintensity)) * 2  # power match
        )

        if score > best_score:
            best_score = score
            best_index = i

    return best_index if best_index != -1 else 0  # default to task 0