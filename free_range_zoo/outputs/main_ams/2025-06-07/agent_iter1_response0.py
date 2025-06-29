def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_supressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
):
    import numpy as np
    
    num_tasks = len(fire_pos)

    # Calculate distances from the agent to each fire
    distances = np.array([np.sqrt((fp[0] - agent_pos[0])**2 + (fp[1] - agent_pos[1])**2) for fp in fire_pos])
    
    # We need to balance the task decision based on several factors:
    # 1. Distance to the fire task
    # 2. Reward weight of the fire task
    # 3. Fire intensity and level

    # Normalize these values
    normalized_distances = np.exp(-distances / 5.0)  # Distance decay factor (temperature of 10.0)
    normalized_intensities = fire_intensities / np.max(fire_intensities + 1e-6)  # Prevent div by zero

    # Calculate potential effectiveness of suppression, which can't be more than available suppressant
    # and it should take into account reduction power per unit suppressant
    possible_suppression = agent_fire_reduction_power * min(agent_supressant_num, 1)

    # Effectiveness scoring model:
    scores = []
    for i in range(num_tasks):
        distance_score = normalized_distances[i]
        weight_score = fire_putout_weight[i] / max(fire_putout_weight)
        intensity_score = 1.0 - normalized_intensities[i]  # prefer more intense fires
        
        # Calculate scope potential: how much a single agent can affect this fire
        # by the available suppressant and fire reduction power
        scope_potential = min(fire_intensities[i], possible_suppression)/fire_intensities[i]
        
        # Integrate all these scores into a final score number
        aggregate_score = weight_score * intensity_score * distance_score * scope_potential
        scores.append(aggregate_score)
    
    # Choosing the task with the highest aggregated score
    selected_task_index = np.argmax(scores)
    
    return selected_task_index