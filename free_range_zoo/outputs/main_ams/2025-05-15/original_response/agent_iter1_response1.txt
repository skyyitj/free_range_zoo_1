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
    Choose the optimal fire-fighting task for a single agent based on fire intensity, distance, and reward weights.
    """
    num_tasks = len(fire_pos)
    maximum_efficient_suppressant_use = 10  # assume maximum efficient suppressant use per task
    
    best_task_score = float('-inf')
    selected_task_index = 0

    distance_normalization_temp = 1/10
    intensity_factor_temp = 0.05
    suppressant_factor_temp = 0.5
    
    for i in range(num_tasks):
        fire_distance = np.hypot(agent_pos[0] - fire_pos[i][0], agent_pos[1] - fire_pos[i][1])
        norm_fire_distance = np.exp(-distance_normalization_temp * fire_distance)
        
        current_intensity = fire_levels[i] * fire_intensities[i]
        norm_intensity = np.exp(-intensity_factor_temp * current_intensity)
        
        suppressant_factor = min(agent_suppressant_num, maximum_efficient_suppressant_use)
        norm_suppressant_usage = np.exp(suppressant_factor_temp * suppressant_factor / agent_suppressant_num)
        
        fire_reduction_potential = agent_fire_reduction_power / (current_intensity + 1)
        
        score = (fire_putout_weight[i] * norm_fire_distance * norm_intensity *
                 norm_suppressant_usage * fire_reduction_potential)
        
        # Compare score to find the best task
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index

# Explanation:
# The policy function takes into consideration the following aspects:
# 1. Distance to the fire (`norm_fire_distance`): Agents prioritize nearer fires.
# 2. Fire intensity (`norm_intensity`): Encourages agents to deal with fires based on their combined level and current intensity value.
# 3. Suppressant usage (`norm_suppressant_usage`): Promotes efficient use of the available fire suppressant.
# 4. Fire reduction potential relative to fire's current intensity (`fire_reduction_potential`).
# The selection score is a composite measure factoring all these elements. The chosen task is the one that maximizes this score, ensuring an optimal balance between resource utilization and effective fire fighting.