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

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    # Parameters to modulate policy sensitivity.
    # Adjusted to achieve better suppressant efficiency and focus on higher intensity tasks.
    level_temperature = 0.50
    intensity_temperature = 0.25
    distance_temperature = 0.05

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    for task in range(num_tasks):

        # Calculate Euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # Calculate score considering fire intensity, agent capabilities, distance and task weight
        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature) * 
            can_put_out_fire * np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) / 
            np.log1p(fire_distance * distance_temperature + 1)
        ) * fire_putout_weight[task] * np.sqrt(agent_suppressant_num)

    max_score_task = np.argmax(scores)
    return max_score_task