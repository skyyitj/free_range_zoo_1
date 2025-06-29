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

    # Calculate distances of the agent to each fire
    distances = np.linalg.norm(
        np.array(fire_pos) - np.array(agent_pos), axis=1
    )

    # Normalize distances to prioritize nearby fires
    inverse_distances = 1 / (distances + 0.01)  # Add a small epsilon to prevent division by zero

    # Scaling factors
    distance_weight = 1.0
    potential_reduction_weight = 2.0
    safety_margin = 0.1  # Encourage not completely depleting suppressant
    
    # Calculate potential reduction in fire intensity after deploying suppressant
    potential_reduction = agent_fire_reduction_power * np.minimum(agent_supressant_num, fire_levels / fire_intensities)

    # Normalize potential reduction
    can_extinguish = potential_reduction >= fire_levels
    
    # Prioritize fire based on a combination of weights and potential reduction
    priority_scores = (
        fire_putout_weight * 
        (distance_weight * inverse_distances + potential_reduction_weight * (np.log(1 + potential_reduction) + 0.5 * can_extinguish))
    )
    
    # Normalize scores
    priority_scores_norm = priority_scores / np.max(priority_scores)
    
    # Account for suppressant conservation -- discourage use of too much suppressant
    suppressant_usage_score = np.log(1 + (agent_supressant_num - potential_reduction) / (agent_supressant_num + safety_margin))
    
    # Combine priority with smart use of resources
    combined_scores = priority_scores_norm + suppressant_usage_score
    
    # Choose the fire with the maximum score
    selected_task_index = np.argmax(combined_scores)

    return selected_task_index