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
    min_distance = np.inf
    closest_fire_index = None
    suppressant_threshold = 10  # set appropriate value

    for idx, fire in enumerate(fire_pos):
        distance = np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2)

        if distance < min_distance:
            min_distance = distance
            closest_fire_index = idx

    # if suppressant is low, opt to recharge instead
    if agent_suppressant_num < suppressant_threshold:
        return -1  # TODO: replace with appropriate index for recharge action

    return closest_fire_index