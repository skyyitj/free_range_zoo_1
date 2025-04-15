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
    Smarter policy that considers suppression capacity, fire levels, and avoids naturally extinguishing fires.
    """
    if not fire_pos or not fire_levels or not fire_intensities:
        return 0  # fallback safe index

    best_score = float('-inf')
    chosen_index = 0  # default

    for i in range(len(fire_pos)):
        level = fire_levels[i]
        intensity = fire_intensities[i]

        # Avoid targeting high fires that would extinguish naturally (assume threshold)
        if level >= 10.0:
            continue

        # Estimate how many agents might also be targeting this
        nearby_agents = sum(
            math.dist(agent_pos, other_pos) < 3.0
            for other_pos in other_agents_pos
        )

        # Compute effectiveness based on how much this agent can suppress
        effective_hit = min(level, agent_fire_reduction_power * agent_suppressant_num)

        # Heuristic score
        distance = math.dist(agent_pos, fire_pos[i]) + 1e-5
        score = (effective_hit / (intensity + 1)) / distance

        # Penalize if many agents are near this fire already
        score -= 0.5 * nearby_agents

        if score > best_score:
            best_score = score
            chosen_index = i

    return chosen_index