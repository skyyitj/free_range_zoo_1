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

    # Improved parameter tuning and scoring methodology
    distance_normalization_temp = 0.03
    intensity_normalization_temp = 0.1

    # Inclusion of agent's current suppressant level into task selection criteria
    available_suppressant_weight = 0.05

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        # Focus on leveraging agent's suppression ability in accord to fire intensity
        effective_suppression_power = min(agent_fire_reduction_power, agent_suppressant_num)
        normalized_effective_power = effective_suppression_power / (1 + fire_intensities[i] * fire_levels[i])
        
        score = fire_putout_weight[i] * normalized_effective_power * norm_distance
        
        # Adjust score based on how much suppressant is left (encourage efficiency)
        suppressant_ratio = 1 - np.exp(-available_suppressant_weight * agent_suppressant_num)
        score *= suppressant_ratio
        
        # Compare score to find the best task
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index