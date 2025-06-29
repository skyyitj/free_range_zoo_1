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

    # Improved temperature parameters for normalization
    remaining_suppressant_temp = 0.1  # More sensitive to suppressant left
    distance_normalization_temp = 0.02  # Less penalty on distance
    intensity_normalization_temp = 0.01  # Less immediate penalty on intensity but focus on high ones

    norm_remaining_suppressant = np.exp(-remaining_suppressant_temp * agent_suppressant_num)

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_normalization_temp * distance)

        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_normalization_temp * intensity)

        # Strengthen the weight of available suppressant and fire level in the score calculation
        score = fire_putout_weight[i] * (agent_fire_reduction_power / (1.0 + intensity)) * norm_distance
        score += np.log1p(fire_levels[i]) * norm_intensity * norm_remaining_suppressant

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index

# This updated policy function adjusts the handling of distances, suppressant availability, and fire intensities.
# These changes are directly designed to improve metrics such as suppressant efficiency, the change in fire intensity,
# and overall rewards. Adjustments to the normalization temperatures were specifically chosen with those goals in mind.