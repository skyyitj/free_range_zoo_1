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

    # Reduce the distance_normalization_temp to give less weight to the distance to the fire, and thus allowing the agent to choose fires that are farther but require more immediate attention.
    distance_normalization_temp = 0.06
    # Increase the intensity_normalization_temp to give more weight to the fire's intensity, and thus prioritize fires with a higher intensity.
    intensity_normalization_temp = 0.12
    
    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        intensity = fire_intensities[i] * fire_levels[i]
        if intensity > 0:
            norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        else:
            norm_intensity = np.exp(-intensity_normalization_temp)
        
        # Slightly decrease the weight given to fire levels to prevent the agent from focusing too much on a single fire, therefore better distributing the suppression resources among different fires.
        score = fire_putout_weight[i] * ((1.0 + agent_suppressant_num / (1.0 + intensity)) * norm_distance * np.sqrt(0.8 * fire_levels[i]) * norm_intensity)
        
        # Compare score to find the best task
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index