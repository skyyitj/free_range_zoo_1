from typing import List, Tuple
import math

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float]
) -> int:
    if not fire_levels:
        return -1  # No fires to consider

    best_score = -math.inf
    best_index = -1

    for i, (pos, level, intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        if level <= 0:
            continue  # Skip extinguished fires

        dy = pos[0] - agent_pos[0]
        dx = pos[1] - agent_pos[1]
        distance = math.sqrt(dy**2 + dx**2)

        # Number of other agents near this fire (within radius)
        nearby_agents = sum(
            math.sqrt((pos[0] - oa[0])**2 + (pos[1] - oa[1])**2) < 2.0
            for oa in other_agents_pos
        )

        # Compute a score: prioritize high intensity, low level, and proximity
        # Penalize if too many agents are nearby
        if agent_suppressant_num > 0:
            score = (
                (intensity / (level + 1)) * agent_fire_reduction_power
                - distance * 0.3
                - nearby_agents * 1.5
            )
        else:
            # No suppressant left: prioritize being near intense fires, but don’t go if covered
            score = (intensity / (level + 1)) - distance * 0.5 - nearby_agents * 2

        if score > best_score:
            best_score = score
            best_index = i

    return best_index if best_index != -1 else 0  # Fallback to 0 to avoid KeyError