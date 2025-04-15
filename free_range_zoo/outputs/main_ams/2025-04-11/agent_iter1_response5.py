import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[float], 
    fire_intensities: List[float]
) -> int:

    num_fires = len(fire_pos)

    # If there's no fire, return -1
    if num_fires == 0:
        return -1

    # Calculate distances to all fires
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]

    # Filter out fires that are too far or the agent doesn't have enough suppressant for.
    valid_fires = [i for i in range(num_fires) if distances[i] <= agent_fire_reduction_power and agent_suppressant_num > 0]

    # If no fires are in range, return -1
    if not valid_fires:
        return -1

    # Otherwise, find the fire with the maximum score calculated by intensity/distance - level
    scores = [(fire_intensities[i]/distances[i] - fire_levels[i]) for i in valid_fires]
    best_fire = valid_fires[np.argmax(scores)]

    return best_fire