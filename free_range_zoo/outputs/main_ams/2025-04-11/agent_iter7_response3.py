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
        return -1  # return -1 to indicate that the agent should recharge

    # Converting agent position to a 2D array for proper subtraction
    agent_pos_2d = np.array(agent_pos)[np.newaxis, :]
    
    # Calculate the distance of each fire from the agent.
    distances_to_fire = np.sqrt(np.sum((np.array(fire_pos) - agent_pos_2d)**2, axis=1))
    
    # Compute reachable fires
    reachable_fires = [(fire_idx, dist) for fire_idx, dist in enumerate(distances_to_fire) if dist <= agent_suppressant_num]

    if not reachable_fires:
        return -1  # return -1 to indicate that the agent should recharge

    # For each fire, calculate the estimated number of suppressant applications required to extinguish it
    suppressant_needed = [(fire_num, fire_levels[fire_num] / agent_fire_reduction_power) for fire_num, _ in reachable_fires]

    # Prioritize fires that need the least suppressant, but scale by inverse of fire intensity
    fire_priorities = [(fire_num, needed / fire_intensities[fire_num])
                       for fire_num, needed in suppressant_needed]

    # Sort the fires by their priority, and select the one that needs the least suppressant to deal with
    fire_priorities.sort(key=lambda x: x[1])
    chosen_fire = fire_priorities[0][0]
    return chosen_fire