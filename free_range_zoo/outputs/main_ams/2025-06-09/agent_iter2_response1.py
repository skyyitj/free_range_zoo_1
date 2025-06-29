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
    best_task_index = -1
    highest_score = float('-inf')
    # Boost resolution for nearly extinguished fires
    close_putout_factor = 10.0

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate distance
        distance = np.sqrt((agent_pos[0] - fire[0]) ** 2 + (agent_pos[1] - fire[1]) ** 2)

        # Simulate potential suppression effects
        if agent_suppressant_num > 0:
            suppressant_use = min(agent_suppressant_num, fire_intensity / agent_fire_reduction_power)
            potential_reduction = suppressant_use * agent_fire_reduction_power
        else:
            potential_reduction = 0
        
        importance_weight = fire_putout_weight[task_index]

        # Factor in how close the fire is to being put out
        urgent_bonus = 0
        if fire_intensity <= agent_fire_reduction_power:
            urgent_bonus = close_putout_factor

        # Adjust temp factors for tuning. Temp controls the sensitivity.
        distance_temp = 2.0  
        reduction_temp = 0.5 
        importance_temp = 1.0
        urgent_temp = 10.0  # High temp to emphasize urgent action when appropriate

        # Scoring function emphasizing quick resolution of close, urgent problems
        task_score = (
            -np.log1p(distance / distance_temp) +  # Disincentivize distance
            np.log1p(potential_reduction / reduction_temp) * (importance_weight / importance_temp) +  # Incentivize fire reduction potential factored by importance
            np.log1p(urgent_bonus / urgent_temp)  # Bonus for nearly extinguished fires
        )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index