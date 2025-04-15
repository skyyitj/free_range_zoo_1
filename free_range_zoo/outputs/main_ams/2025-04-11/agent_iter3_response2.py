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
    
    if agent_suppressant_num <= 0:
        return -1
    
    scores = []
    
    for task_id in range(num_fires):
        distance =(np.sqrt((fire_pos[task_id][0]-agent_pos[0])**2 + (fire_pos[task_id][1]-agent_pos[1])**2))
        score = fire_levels[task_id]*fire_intensities[task_id]/(distance*agent_fire_reduction_power*agent_suppressant_num)
        scores.append(score)
    
    min_distance_task = np.argmax(scores)

    return min_distance_task