def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float, 

    other_agents_pos: List[Tuple[float, float]], 

    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float], 

    fire_putout_weight: List[float]
) -> int:
    # Compute euclidean distance to fires
    distances_to_fires = [np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2) 
                        for i in range(len(fire_pos))]

    # Set priority based on distance, fire intensity and fire level
    priorities = [(0.4 / distances_to_fires[i]) + (0.3 * fire_intensities[i]) + (0.3 * fire_levels[i])
                        for i in range(len(fire_pos))]

    # Assign agent to the fire with the highest priority
    assigned_fire = np.argmax(priorities)

    return assigned_fire