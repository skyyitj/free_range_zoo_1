def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Position of the agent (y, x)
    agent_fire_reduction_power: float,           # Fire suppression power
    agent_suppressant_num: float,                # Fire suppressant resources

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Positions of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value for each fire

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for suppression tasks
) -> int:

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Temperatures
    level_temperature = 0.45 # Increased level_temperature to prioritize intense fire
    intensity_temperature = 0.25 # Increased intensity_temperature to tackle intense fire
    distance_temperature = 0.05 # Decreased distance_temperature to focus less on distance but more on fire levels

    for task in range(num_tasks):

        # Calculate Euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # Use fire level, intensity, and distance to calculate score for each task
        # Increase impact of agent_supressant_num and fire_putout_weight
        # Score is the sum of the exponential of negative weighted components
        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature) +
            can_put_out_fire * np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) +
            # Changed from subtracting the fire_distance to adding it to balance 'fire_distance' and 'agent_suppressant_num'
            fire_distance * np.exp(-fire_distance * distance_temperature)
        ) * fire_putout_weight[task] * np.log(agent_suppressant_num+1) # Changed np.sqrt(agent_suppressant_num) to np.log(agent_suppressant_num+1) for prudent resource usage

    # Return task index with maximum score
    max_score_task = np.argmax(scores)
    return max_score_task