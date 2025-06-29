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
    fire_levels: List[int],                    # Current intensity level of each fire
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

    distance_normalization_temp = 0.03  # Decrease the temperature scale for distance normalization
    intensity_normalization_temp = 0.01  # Decrease the temperature scale for intensity normalization
    agent_fire_reduction_power = min(agent_fire_reduction_power, agent_suppressant_num)
    
    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        intensity = fire_intensities[i] * fire_levels[i]
        if intensity > 0:
            norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        else:
            norm_intensity = np.exp(-intensity_normalization_temp)
        
        score = fire_putout_weight[i] * (agent_fire_reduction_power / (1.0 + intensity)) * norm_distance * norm_intensity
        
        # Compare score to find the best task
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index

# Improvement Changes:
# 1. Adjusted the temperature of the normalization term for the distance from 0.1 to 0.03 to make agents more aware of their positions and make more informed decisions.
# 2. Adjusted the temperature of the normalization term for the fire intensity from 0.05 to 0.01. This change makes the agents more sensitive to the intensity of the fire, motivating them to tackle the fires more effectively.
# 3. Modified the agent's fire reduction power to consider the available suppressant. This encourages the agent to use its resources wisely and to choose fires that it can control effectively with the available suppressant.
# The modifications aim to improve the efficiency of fire control and resource utilization. The tuning of the temperature parameters is designed to strike a balance between agent mobility and fire fighting efficiency. By adjusting the fire reduction power to consider the available suppressant, we hope to maximize the use of resources and minimize wastage.