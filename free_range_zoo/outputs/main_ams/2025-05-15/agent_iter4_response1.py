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

    # Temperature adjustment and trimming for policy tuning
    distance_temperature = 0.009   # Lowering more to allow flexibility in firefighting ranges
    intensity_temperature = 0.01   # Reducing to balance the firefighting between highly intense fires and regular fires
    suppressant_conserve_factor = 15.0   # Boosting the suppressant preservation

    remaining_suppressant = np.exp(-suppressant_conserve_factor * (1 - (agent_suppressant_num / 10)))
    resource_proximity_balance_factor = 0.7  # Introducing a balancing factor to weigh the resource conservation and firefighting efforts

    for i in range(num_tasks):
        # Calculating Euclidean distance
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)
        
        # Normalizing fire intensity
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * intensity)

        # Resource conservation and proximity-weighted firefighting score
        firefighting_score = fire_putout_weight[i] * (resource_proximity_balance_factor * norm_distance * agent_fire_reduction_power / (1 + intensity)) * np.log1p(fire_levels[i]) * remaining_suppressant * norm_intensity

        if firefighting_score > best_task_score:
            best_task_score = firefighting_score
            selected_task_index = i

    return selected_task_index