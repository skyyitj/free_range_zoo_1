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

    if agent_suppressant_num <= 0:
        # recharge if out of suppressant
        return -1

    # Create a list of tuples containing fires that are within the agent's reach
    reachable_fires = [(i, np.sqrt(((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2)))
                       for i, fire in enumerate(fire_pos) 
                       if np.sqrt(((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2)) <= agent_suppressant_num]

    if not reachable_fires:
        # no fires can be extinguished with the current amount of suppressant, it's better to recharge
        return -1  # return -1 or a special value to indicate that the agent should recharge

    # For each fire, calculate the estimated number of suppressant applications required to extinguish it
    suppressant_needed = [(fire_num, fire_levels[fire_num] / agent_fire_reduction_power) for fire_num, _ in reachable_fires]

    # Prioritize fires that need the least suppressant, but scale by inverse of fire intensity 
    # (we want to prioritize higher intensity fires IF we have the suppressant to deal with them)
    fire_priorities = [(fire_num, needed / fire_intensities[fire_num])
                       for fire_num, needed in suppressant_needed]

    # Sort the fires by their priority, and select the one that needs the least suppressant to deal with
    fire_priorities.sort(key=lambda x: x[1])
    chosen_fire = fire_priorities[0][0]
    return chosen_fire