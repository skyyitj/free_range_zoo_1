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
    
    # If no suppressant, do nothing
    if agent_suppressant_num <= 0:
        return -1
    
    # Calculate the distance to each fire
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]
    
    # Calculate score for each fire considering distance, intensity and agents' fire reduction power
    fire_scores = [(dist/(fire_intensities[i]*agent_fire_reduction_power*agent_suppressant_num)) + fire_intensities[i] for i, dist in enumerate(distances)]
    
    # Choose the fire with lowest score (easiest to extinguish with available resources)
    chosen_fire = np.argmin(fire_scores)
    
    return chosen_fire