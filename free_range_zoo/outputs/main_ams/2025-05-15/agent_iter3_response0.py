import numpy as np

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
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    # Updated parameters for more aggressive and resource-aware policy:
    remaining_suppressant_temp = 0.05  # Increased consideration for remaining suppressant
    distance_normalization_temp = 0.03  # Slightly adjusted for balancing
    intensity_normalization_temp = 0.07  # More aggressive on intensity

    norm_remaining_suppressant = np.exp(-remaining_suppressant_temp * max(0, agent_suppressant_num - 250))  # Consider effective suppressant left

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_normalization_temp * distance)

        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_normalization_temp * intensity)

        # Score is recalculated with subtly changed weights
        score = fire_putout_weight[i] * (agent_fire_reduction_power / (1.0 + intensity) * norm_distance * np.sqrt(fire_levels[i]))
        score *= norm_intensity * norm_remaining_suppressant

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index