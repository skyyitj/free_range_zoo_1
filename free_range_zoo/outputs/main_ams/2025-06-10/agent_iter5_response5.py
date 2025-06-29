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
    task_scores = []

    for i in range(num_tasks):
        # Calculate distance from agent to fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        distance_weight = np.exp(-distance / 5)  # Normalizing distance effect with a temperature parameter (5)

        # Calculate the amount of fire that can be potentially suppressed
        obtainable_suppression = min(agent_suppressant_num * agent_fire_reduction_power, fire_intensities[i])
        suppression_weight = np.log(1 + obtainable_suppression)  # Logarithmic transformation with a natural base

        # Final task weight
        # The higher the fire intensity and priority weight, the more attractive the task should be
        task_weight = distance_weight * suppression_weight * fire_putout_weight[i]
        task_scores.append(task_weight)

    # Select the task with the highest score
    best_task_index = np.argmax(task_scores)
    return best_task_index