from typing import Tuple, List

def single_agent_policy(
    agent_pos: Tuple[float, float],  # Current position of the agent (y, x)
    agent_fire_reduction_power: float,  # How much fire the agent can reduce
    agent_suppressant_num: float,  # Amount of fire suppressant available
    other_agents_pos: List[Tuple[float, float]],  # Positions of all other agents
    fire_pos: List[Tuple[float, float]],  # Locations of all fires
    fire_levels: List[int],  # Current intensity level of each fire
    fire_intensities: List[float],  # Current intensity of each fire
    fire_putout_weight: List[float]  # Priority weights for fire suppression tasks
) -> int:
    import numpy as np
    
    # Constants for score calculations
    distance_factor_temp = 1.0
    suppressant_factor_temp = 1.0
    intensity_factor_temp = 0.5
    level_factor_temp = 0.5
    weight_factor_temp = 0.1
    
    num_tasks = len(fire_pos)
    scores = []
    
    # Compute and collect scores for each fire
    for i in range(num_tasks):
        # Calculate the Euclidean distance from agent to fire
        distance = np.sqrt(
            (agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        
        # Normalize intensity and levels considering the agent's power and available suppressant
        effective_intensity = fire_intensities[i] * fire_levels[i]
        potential_suppress = agent_suppressant_num * agent_fire_reduction_power
        
        # Score attributes
        distance_score = np.exp(-distance * distance_factor_temp)
        intensity_score = np.exp(-effective_intensity * intensity_factor_temp)
        suppressant_score = np.exp(min(potential_suppress/effective_intensity, 1) * suppressant_factor_temp)
        level_score = np.exp(-fire_levels[i] * level_factor_temp)
        weighted_priority = fire_putout_weight[i] * weight_factor_temp
        
        # Composite score combining all attributes
        score = (distance_score + intensity_score + suppressant_score + level_score) * weighted_priority
        scores.append(score)
    
    # Select the task with the highest score
    selected_task_index = np.argmax(scores)
    return selected_task_index

# Example usage (the real testing would use suitable realistic data)
# sample_data = {
#    "agent_pos": (0.5, 0.5),
#    "agent_fire_reduction_power": 2.0,
#    "agent_suppressant_num": 100,
#    "other_agents_pos": [(2, 2), (3, 3)],
#    "fire_pos": [(1, 1), (4, 4)],
#    "fire_levels": [3, 5],
#    "fire_intensities": [1.0, 1.5],
#    "fire_putout_weight": [2.0, 3.0]
# }
# print(single_agent_policy(**sample_data))  # Expected output: index of the selected task (e.g., 0 or 1)