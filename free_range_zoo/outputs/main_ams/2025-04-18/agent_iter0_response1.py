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
    fire_putout_weight: List[float],
) -> int:
    # Initialize an empty list to store scores for each task
    task_scores = []

    distance_temperature = 0.01
    resource_consumption_temperature = 0.1
    intensity_temperature = 0.05
    weight_temperature = 1.0

    # Loop through each task to calculate its score
    for i in range(len(fire_pos)):
        # Calculate the distance between the agent and the fire task
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)

        # Consider the agent's suppressant consumption for this task
        suppressant_consumption = min(agent_fire_reduction_power, agent_suppressant_num)

        # Incorporate the intensity of the fire
        intensity = fire_intensities[i]

        # Normalize and weigh components of the task score
        normalized_distance_score = np.exp(-distance * distance_temperature)
        normalized_resource_score = np.exp(-suppressant_consumption * resource_consumption_temperature)
        normalized_intensity_score = np.exp(-intensity * intensity_temperature)
        weight_score = np.exp(fire_putout_weight[i] * weight_temperature)

        # Combine task components into a comprehensive score
        task_score = normalized_distance_score * normalized_resource_score * normalized_intensity_score * weight_score

        task_scores.append(task_score)

    # Return the task index with the highest score
    return np.argmax(task_scores)