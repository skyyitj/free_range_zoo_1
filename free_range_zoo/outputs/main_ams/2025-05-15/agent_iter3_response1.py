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

    distance_temp = 0.1
    intensity_temp = 0.2
    resource_temp = 0.2
    weight_temp = 0.3

    # Normalize resources based on the average amount needed when fully combating a fire
    avg_intensity_expected = np.mean(fire_intensities)
    avg_level_expected = np.mean(fire_levels)
    est_avg_resources_needed = avg_intensity_expected * avg_level_expected / agent_fire_reduction_power

    estimated_suppressant_consumed = agent_suppressant_num / max(1, est_avg_resources_needed)
    norm_suppressant_reserves = np.exp(-resource_temp * (1 - estimated_suppressant_consumed))

    for i in range(num_tasks):
        fire_location = fire_pos[i]
        distance = np.sqrt((agent_pos[0] - fire_location[0])**2 + (agent_pos[1] - fire_location[1])**2)
        norm_distance = np.exp(-distance_temp * distance)
        
        fire_intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temp * fire_intensity)

        # Use the weight directly but allow for exponential scaling to prioritize higher weights more
        task_weight = fire_putout_weight[i]
        prioritized_task_weight = np.exp(weight_temp * task_weight)

        # The task score includes all factors, higher score = better task to select
        score = prioritized_task_weight * norm_distance * norm_intensity * norm_suppressant_reserves

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index

# The revised policy takes into consideration the suppressant resource more accurately:
# 1. Introduces normalization of suppressant reserves based on estimated average resources needed.
# 2. Influence of distance and intensity is modulated using temperature values.
# 3. Tasks weights are influenced exponentially, emphasizing the importance of task priority provided by weights.
# This policy should better balance the resource conservation against the effectiveness of fire suppression by integrating a more informed estimation of suppressant consumption.