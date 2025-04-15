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

    # Calculate the expected suppressant usage of all fire tasks
    expected_suppressant_usage = [fire_levels[i]/agent_fire_reduction_power for i in fires_within_reach]

    # Calculate the "value" of each fire task: fire intensity * expected suppressant usage
    fire_values = [fire_intensities[i] * suppressant for i, suppressant in zip(fires_within_reach, expected_suppressant_usage)]
    # Calculate the distances from other agents to each fire task
    distances_from_other_agents = np.asarray([[np.sqrt((fire[0] - agent[0]) ** 2 + (fire[1] - agent[1]) ** 2)
                                               for fire in fire_pos] for agent in other_agents_pos])

    # Adjust the values of the fire tasks based on the distances of other agents
    for i, fire_value in enumerate(fire_values):
        closest_agent_distance = min(distances_from_other_agents[:, i])
        fire_values[i] = fire_value * (1 - closest_agent_distance / max(distances_from_other_agents[:, i]))

    # Choose the fire task with the highest adjusted value
    chosen_fire = max(range(len(fire_values)), key=fire_values.__getitem__)
    return chosen_fire