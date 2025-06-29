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

    distance_normalization_temp = 0.08   # Decreased the temperature scale to better balance the task score components
    intensity_normalization_temp = 0.07

    # Adjacent agent positions
    adjacent_agent_pos = [pos for pos in other_agents_pos if np.sqrt((agent_pos[0] - pos[0]) ** 2 + (agent_pos[1] - pos[1]) ** 2) <= 1]
    
    for i in range(num_tasks):
        
        # More weightage given to tasks at locations where there are fewer defenders
        num_adjacent = len([pos for pos in adjacent_agent_pos if np.sqrt((fire_pos[i][0] - pos[0]) ** 2 + (fire_pos[i][1] - pos[1]) ** 2) <= 1])
        defence_factor = 1/(num_adjacent + 1)
        
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        intensity = fire_intensities[i] * fire_levels[i]
        if intensity > 0:
            norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        else:
            norm_intensity = np.exp(-intensity_normalization_temp)
        
        score = fire_putout_weight[i] * (agent_fire_reduction_power / (1.0 + intensity)) * norm_distance * np.sqrt(fire_levels[i]) * norm_intensity * defence_factor
        
        # Compare score to find the best task
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index

# The policy function has been improved by taking into account the distribution of the other agents.
# The revised policy adds a 'defence_factor' to the task score to prioritize tasks at locations where there are fewer defenders. This distributes the agents to cover more tasks efficiently.