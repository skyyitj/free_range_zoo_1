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
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    # Modify the temperature parameters
    remaining_suppressant_temp = 0.03  # Adjusted from 0.05
    distance_normalization_temp = 0.02  # Adjusted from 0.05, less penalization on distance
    intensity_normalization_temp = 0.02  # Adjusted from 0.05, less penalization on intensity
    
    # Calculate normalized suppressant remaining
    if agent_suppressant_num > 0:
        norm_remaining_suppressant = np.exp(-remaining_suppressant_temp / agent_suppressant_num)
    else:
        norm_remaining_suppressant = 0

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_normalization_temp * distance)

        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_normalization_temp * intensity)

        # Consider fire intensity and firepower ratio to see if the agent can manage the fire
        firepower_ratio = agent_fire_reduction_power / (1.0 + intensity)
        
        # Incorporate the remaining suppressant into fire management calculations
        suppressant_cost = min(agent_suppressant_num, intensity / agent_fire_reduction_power)
        suppressant_efficiency = firepower_ratio / suppressant_cost if suppressant_cost > 0 else 0
        
        # Calculate the score with updated factors and considerations
        score = (fire_putout_weight[i] * norm_distance *
                 norm_intensity * np.sqrt(fire_levels[i]) *
                 norm_remaining_suppressant * suppressant_efficiency)

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index

# The updates aim to improve resource management by directly incorporating suppressant costs and suppressant efficiency into the task scoring. Additionally, adjustments in temperature parameters lead to less severe penalties on distance and intensity, potentially allowing agents to address higher-intensity or farther fires when strategically necessary. These changes aim to directly address the efficiency and suppressant utilization metrics, while also balancing the task prioritization with the adjusted scoring mechanics.