def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], 

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                      
    fire_intensities: List[float],               

    # === Task Prioritization ===
    fire_putout_weight: List[float],             
) -> int:

    # Calculate the euclidian distance 
    def distance(coord1, coord2):                   
        return ((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)**0.5 
        #sqrt(pow((coord1.getX()-coord2.getX()),2) + pow((coord1.getY()-coord2.getY()),2))

    num_tasks = len(fire_pos)
    task_scores = np.zeros(num_tasks)

    for i in range(num_tasks):
        task_dist = distance(agent_pos, fire_pos[i])  # Distance from agent to task
        task_intensity = fire_intensities[i]          # Fire intensity of task
        task_level = fire_levels[i]                   # Fire level of task
        
        # Prioritize task based on distance, intensity, and available resources
        task_score = (agent_suppressant_num / task_dist) * task_intensity * agent_fire_reduction_power * fire_putout_weight[i]

        # Penalize task score if other agents are closer
        for other_agent_pos in other_agents_pos:
            if distance(other_agent_pos, fire_pos[i]) < task_dist:
                task_score *= 0.5

        task_scores[i] = task_score

    # Get task with highest score
    best_task = np.argmax(task_scores)
    
    return best_task