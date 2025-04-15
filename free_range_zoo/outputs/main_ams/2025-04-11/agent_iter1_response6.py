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
   
    if num_fires == 0:
        return None

    # Calculate distances to all fires
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]

    # Initialize the best task id and its associated highest score
    best_task = -1
    highest_score = -np.inf

    # For each fire
    for task_id in range(num_fires):
        if agent_suppressant_num <= 0:
            break

        task_score = (fire_intensities[task_id] 
                      * agent_fire_reduction_power 
                      * agent_suppressant_num
                      / distances[task_id]
                      - fire_levels[task_id])

        if task_score > highest_score:
            highest_score = task_score
            best_task = task_id

        agent_suppressant_num -= 1

    return best_task