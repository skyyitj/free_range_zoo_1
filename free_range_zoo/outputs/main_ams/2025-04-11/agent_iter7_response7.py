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
      # recharging suppressant if out of suppressant.
      return -1

    # Compute distances from agent to all fire positions.
    distances_to_fire = np.sqrt(np.sum((np.array(fire_pos) - np.array(agent_pos))**2, axis=1))

    # If agent can not reach any firefighting task, recharge suppressant as closer to the one can be reached soon.
    reachable_tasks = np.where(agent_suppressant_num - distances_to_fire > 0, True, False)
    if np.sum(reachable_tasks) == 0:
        return np.argmin(distances_to_fire)

    # Compute the "attack power" of the agent towards each fire, as the product of the suppressant within the distance limit and the fire reduction power per suppressant.
    attack_powers = (agent_suppressant_num - distances_to_fire)*agent_fire_reduction_power

    # Normalize fire intensities.
    normalized_intensities = (fire_intensities - np.min(fire_intensities)) / (np.max(fire_intensities) - np.min(fire_intensities))

    # We combine these information to form "priorities" of each firefighting tasks, as the product of attack power and fire intensity.
    priorities = attack_powers * (1 - normalized_intensities)
    priorities[~reachable_tasks] = -np.inf  # Assign the tasks that are unreachable with the current suppressant level a priority of negative infinity.

    # Choose the task with the maximum priority.
    return np.argmax(priorities)