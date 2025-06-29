import numpy as np

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

    remaining_suppressant_temp = 0.05
    distance_normalization_temp = 0.05  
    intensity_normalization_temp = 0.05

    norm_remaining_suppressant = np.exp(-remaining_suppressant_temp * agent_suppressant_num)

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_normalization_temp * distance)

        intensity = fire_intensities[i] * fire_levels[i]
        if intensity > 0:
            norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        else:
            norm_intensity = np.exp(-intensity_normalization_temp)

        score = fire_putout_weight[i] * (agent_fire_reduction_power / (1.0 + intensity)) * norm_distance * np.sqrt(fire_levels[i]) * norm_intensity
        score *= norm_remaining_suppressant

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index

# Given the metric results, we might consider adjusting the policy.
# Based on the results, the policy isn't performing as well. We might want to incorporate the remaining suppressant into our task scores.
# This will give preference to those fires which can be handled without completely depleting agent's resources, thus ensuring better overall performance. 
# Also, introducing a new temperature "remaining_suppressant_temp" related with the remaining suppressant not only helps the agent to better balance between short-term benefits and long-term performance, but also improves overall use efficiency. 
# In the end, we may adjust the normalization_temps for distance and intensity to achieve better performance. This is because the proper normalization_temps will help the agent better balance between the short-term benefits and the long-term performance. A too large temp will lead the agent to blindly pursue short-term benefits, while a too small temp will cause the agent to struggle to make progress. Adjusting the temperature to an appropriate value can help the agent balance these two aspects and then achieve a good performance. 
# Consequently, the improved policy function would look as above.