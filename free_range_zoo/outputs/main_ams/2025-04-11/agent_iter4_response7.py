import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    distances = [np.sqrt((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2) for fire in fire_pos]
    fires_within_reach = [i for i, dist in enumerate(distances) if dist <= agent_suppressant_num/agent_fire_reduction_power]

    if not fires_within_reach:
        # no fires can be extinguished with the current amount of suppressant, it's better to recharge
        return -1  # return -1 or a special value to indicate that the agent should recharge

    # Otherwise, extinguish the most intense fire that's closest to the agent
    chosen_fire = max(fires_within_reach, key=lambda i: (fire_levels[i], fire_intensities[i]))
    return chosen_fire