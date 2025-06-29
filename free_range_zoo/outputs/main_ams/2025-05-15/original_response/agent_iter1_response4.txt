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

    distance_normalization_temp = 0.01
    intensity_normalization_temp = 0.05
    fire_reducing_power = agent_fire_reduction_power * agent_suppressant_num
    
    for i in range(num_tasks):
        # Calculate the Euclidean distance between agent and fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        # Calculate the probable intensity after fire reducing
        intensity = max(0, fire_intensities[i] * fire_levels[i] - fire_reducing_power)
        norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        
        # Calculate the score for the task
        score = fire_putout_weight[i] * norm_distance * norm_intensity
        
        # Compare score to find the best task
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index