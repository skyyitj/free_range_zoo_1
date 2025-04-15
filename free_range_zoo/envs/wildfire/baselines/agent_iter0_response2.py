def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],

    # Task information
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    if agent_suppressant_num <= 0: 
        # If the agent has no suppressant left, return -1 so that it can move to recharge.
        return -1

    # Calculate distances from the agent to all fires.
    distances = [((fx-agent_pos[0])**2 + (fy-agent_pos[1])**2)**0.5 for fx, fy in fire_pos]

    # Create a list to store (index, distance) pairs where the fire isn't self-extinguishing
    non_extinguishing_fires = [(i, d) for i, d in enumerate(distances) if fire_levels[i] < fire_intensities[i]]

    if not non_extinguishing_fires:
        # If all fires are self-extinguishing, return -1.
        return -1

    # Find the fires with minimum distance
    min_distance = min(non_extinguishing_fires, key=lambda x:x[1])[1]
    closest_fires = [i for i, d in non_extinguishing_fires if d == min_distance]

    # If there are multiple fires with the same minimum distance, find the one with maximum intensity.
    if len(closest_fires) > 1:
        max_intensity = max([fire_intensities[i] for i in closest_fires])
        task_to_address = [i for i in closest_fires if fire_intensities[i] == max_intensity][0]
    else:
        task_to_address = closest_fires[0]

    return task_to_address