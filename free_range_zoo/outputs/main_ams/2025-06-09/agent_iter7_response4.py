def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float,
    agent_supressant_num: float, 
    other_agents_pos: List[Tuple[float, float]], 
    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    num_tasks = len(fire_pos)
    best_task_index = -1
    highest_score = float('-inf')

    # Adjust the temperature constants for score weighting
    distance_temp = 0.4  # Increase the sensitivity to distance even more
    effectiveness_temp = 4.0  # Increase emphasis on suppressing high intensity fires
    importance_temp = 4.0  # Increase emphasis on fires with higher reward weights
    suppressant_eff_temp = 3.0  # Higher emphasis on efficient use of suppressants

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate Euclidean distance to each fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Calculate the maximum effective suppressant use
        target_suppressant_use = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_supressant_num)
        
        # Effective reduction in fire intensity this agent can achieve
        potential_effectiveness = agent_fire_reduction_power * target_suppressant_use
        
        # Calculate suppressant efficiency (avoid division by zero)
        if target_suppressant_use > 0:
            suppressant_efficiency = potential_effectiveness / target_suppressant_use
        else:
            suppressant_efficiency = 0 
        
        importance_weight = fire_putout_weight[task_index]
        
        # Calculate the task score factoring all components
        task_score = (
            -np.exp(distance / distance_temp)  # Prioritize closer fires significantly
            + np.log1p(potential_effectiveness) * effectiveness_temp  # Emphasize extinguishing effectiveness
            + np.log1p(suppressant_efficiency) * suppressant_eff_temp  # Value efficient use of suppressants
            + np.exp(importance_weight) * importance_temp  # More bias towards fires with higher importance weights
        )
        
        # Select the task with the maximum score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index