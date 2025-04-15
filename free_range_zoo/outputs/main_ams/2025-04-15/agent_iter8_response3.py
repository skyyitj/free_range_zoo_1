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
    fire_putout_weight: List[float]
) -> int:

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)
    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Temperatures
    other_agents_distance_temp = 0.2
    level_temp = 0.6
    intensity_temp = 0.3
    distance_temp = 0.4
    agent_suppressant_temp = 0.4

    for task in range(num_tasks):
        min_other_agent_distance = min([distance.euclidean(other_agent_pos, fire_pos[task]) for other_agent_pos in other_agents_pos])
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        scores[task] = (
            np.exp(other_agents_distance_temp * min_other_agent_distance) +
            (fire_putout_weight[task]) * np.exp((fire_levels[task] / can_put_out_fire) * -level_temp) +
            can_put_out_fire * np.exp(fire_intensities[task] / can_put_out_fire * -intensity_temp) -
            fire_distance * np.exp(fire_distance * -distance_temp) +
            (agent_suppressant_num / fire_levels[task]) * np.exp((fire_levels[task] / agent_suppressant_num) * agent_suppressant_temp)
        )

    # Return task index with maximum score
    max_score_task = np.argmax(scores)
    return max_score_task