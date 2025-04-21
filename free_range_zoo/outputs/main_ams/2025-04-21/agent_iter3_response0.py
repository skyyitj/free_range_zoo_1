def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float],
) -> int:
    num_fires = len(fire_pos)

    # Adjusting temperature variables based on the policy evaluation feedback
    distance_temp = 0.2  # high value to favor closer fire.
    intensity_temp = 1  # low value to penalize higher intensity fire.
    weight_temp = 2

    task_scores = []
    for i in range(num_fires):

        # Calculate Euclidean distance to the fire
        distance = math.sqrt(
            (agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2
        )
        # Normalize distance using exponential function
        exp_distance = np.exp(-distance / distance_temp)
        
        # Normalize fire intensity
        exp_intensity = np.exp(-fire_intensities[i] / intensity_temp)
        
        # Consider fire's priority weight
        exp_weight = np.exp(fire_putout_weight[i] / weight_temp)

        # Calculate task score
        # Increasing the weight for remaining suppressant to 2
        task_score = (
            exp_distance * exp_intensity * exp_weight * (2 + agent_suppressant_num)
        )
        task_scores.append(task_score)

    # Return the index of the fire task with maximum score
    return task_scores.index(max(task_scores))