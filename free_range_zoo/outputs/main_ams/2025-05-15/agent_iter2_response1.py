import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    distance_normalization_temp = 0.1  # Smaller values if focusing on closer fires is too dominating
    intensity_normalization_temp = 0.025  # Adjust for more aggressive action on high intensity fires
    agent_effectiveness_temp = 1.5  # High value to emphasize more on agent's current effectiveness

    for i in range(num_tasks):
        # Calculate the distance to each fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        # Check the intensity of each fire
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        
        # Compute effectiveness score after agent's suppressant application
        potential_intensity_after_action = max(0, intensity - agent_fire_reduction_power * agent_suppressant_num)
        potential_norm_intensity_after_action = np.exp(-intensity_normalization_temp * potential_intensity_after_action)
        effectiveness_score = (norm_intensity - potential_norm_intensity_after_action) / (1 + distance)

        # Weight with the priority of the task
        score = fire_putout_weight[i] * effectiveness_score * np.power(agent_fire_reduction_power, agent_effectiveness_temp)
        
        # Compare score to find the best task
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index