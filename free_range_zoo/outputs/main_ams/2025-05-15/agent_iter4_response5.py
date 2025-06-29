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

    # Fine-tuning temperature and factor parameters
    distance_temperature = 0.1  # Moderate emphasis on distance
    intensity_temperature = 0.05  # Enhanced emphasis on fire intensity
    suppressant_preservation_factor = 5.0  # Balance between use and preservation

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        fire_importance = (fire_levels[i] * fire_intensities[i])
        norm_distance = np.exp(-distance_temperature * distance)
        norm_fire_imp = np.exp(intensity_temperature * fire_importance)

        expected_fire_reduction = agent_fire_reduction_power / (1 + fire_importance)
        suppressant_usage_effectiveness = np.exp(-suppressant_preservation_factor * expected_fire_reduction)

        # Compute task score with new sensitivity parameters
        score = fire_putout_weight[i] * norm_distance * norm_fire_imp * suppressant_usage_effectiveness

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index

# This updated policy now better balances suppression urgency with resource efficiency and task prioritization.