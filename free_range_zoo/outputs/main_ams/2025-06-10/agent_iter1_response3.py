import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]],

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],

    # === Task Prioritization ===
    fire_putout_weight: List[float]
) -> int:
    num_tasks = len(fire_pos)
    scores = np.zeros(num_tasks)

    for i in range(num_tasks):
        # Distance factor
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))
        inv_distance = 1.0 / (distance + 0.1)  # Adding a small constant to prevent division by zero

        # Fire intensity factor
        remaining_fire_intensity = max(0, fire_intensities[i] - agent_suppressant_num * agent_fire_reduction_power)
        fire_intensity_factor = 1 / (remaining_fire_intensity + 1)  # Adding 1 to normalize

        # Reward weighting factor
        reward_factor = fire_putout_weight[i]
        
        # Combining factors for task score
        score = reward_factor * inv_distance * fire_intensity_factor
        
        scores[i] = score

    # Choose task with maximum score
    selected_task_index = np.argmax(scores)
    return selected_task_index