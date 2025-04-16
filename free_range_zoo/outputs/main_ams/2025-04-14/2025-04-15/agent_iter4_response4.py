def single_agent_policy_v2(
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
    
    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Adjusted temperature parameters
    level_temperature_v2 = 0.3
    intensity_temperature_v2 = 0.1
    distance_temperature_v2 = 0.08

    for task in range(num_tasks):

        # get euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # calculate score for each task using fire intensity, level, distance and weight for priority
        # all values are scaled to reward efficient resource allocation and fire suppression
        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature_v2) +
            can_put_out_fire * np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature_v2) -
            fire_distance * np.exp(fire_distance * distance_temperature_v2)
        ) * fire_putout_weight[task]

    # assign agent to highest score task
    max_score_task = np.argmax(scores)
    return max_score_task