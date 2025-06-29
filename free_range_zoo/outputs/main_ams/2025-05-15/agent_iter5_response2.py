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

    # Parameters for policy tuning
    distance_temperature = 0.01  # Increase importance of distance
    intensity_temperature = 0.05  # More focus on high intensity fires
    resource_management_factor = 20.0  # Less aggressive suppressant usage

    remaining_suppressant_factor = np.exp(-resource_management_factor * (1 - (agent_suppressant_num / 10)))

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)

        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * intensity)

        # Adjusted score computation with revised weighting
        score = fire_putout_weight[i] * (norm_distance * agent_fire_reduction_power / (1 + intensity)) * np.log1p(fire_levels[i]) * remaining_suppressant_factor * norm_intensity

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index