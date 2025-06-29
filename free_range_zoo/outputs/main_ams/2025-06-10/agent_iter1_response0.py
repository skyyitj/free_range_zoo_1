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
    import numpy as np

    num_fires = len(fire_pos)
    scores = []

    # Constants for score computation
    distance_scale = 1.0
    level_scale = 2.0
    intensity_scale = 3.0
    weight_scale = 5.0
    suppressant_scale = 4.0

    for i in range(num_fires):
        # Calculate Euclidean distance from agent to fire
        dist = np.sqrt((fire_pos[i][0] - agent_pos[0]) ** 2 + (fire_pos[i][1] - agent_pos[1]) ** 2)

        # Direct factors
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        task_weight = fire_putout_weight[i]

        # Anticipated fire reduction (consider agent's current suppressant amount and its power)
        expected_reduction = min(agent_suppressant_num * agent_fire_reduction_power, fire_intensity)

        # Compute Score for this task
        # Here we are prioritizing:
        # - Lower distances
        # - Higher fire levels (more urgent to control)
        # - Higher fire intensities (bigger fires)
        # - Lower agent suppression resource usage (encourage saving resources)
        # - Higher fire put out weights (priority fires)
        score = (
            (-distance_scale * np.log(dist + 1)) + 
            (level_scale * np.log(fire_level + 1)) + 
            (intensity_scale * np.log(fire_intensity + 1)) - 
            (suppressant_scale * np.log(expected_reduction + 1)) + 
            (weight_scale * task_weight)
        )

        scores.append(score)

    # Choose the fire task with the highest score
    chosen_task_index = int(np.argmax(scores))

    return chosen_task_index