from typing import List, Tuple
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
    # Calculate the distance of each fire from the agent
    distances = [np.sqrt((fp[0] - agent_pos[0])**2 + (fp[1] - agent_pos[1])**2) for fp in fire_pos]

    # Normalize distances to use in score calculation
    max_distance = np.max(distances) if distances else 1  # Avoid division by zero
    normalized_distances = [d / max_distance for d in distances]

    # Potential effectiveness of suppression - assuming limited resources
    potential_effectiveness = [min(agent_suppressant_num * agent_fire_reduction_power / intensity, 1) 
                               for intensity in fire_intensities]

    # Score calculation based on task properties
    scores = []
    distance_temp = 0.5  # Temperature parameter for distance conversion
    effectiveness_temp = 1.5  # Temperature parameter for effectiveness conversion
    for i in range(len(fire_pos)):
        # Scores are calculated based on normalized distance (inversely proportional), potential effectiveness, and task weight
        # Use exponential function to transform scores and emphasize differences
        score = np.exp(-normalized_distances[i] / distance_temp) * \
                np.exp(potential_effectiveness[i] / effectiveness_temp) * \
                fire_putout_weight[i]
        scores.append(score)
    
    # Choose the task with the highest score
    best_task_index = np.argmax(scores)
    return best_task_index