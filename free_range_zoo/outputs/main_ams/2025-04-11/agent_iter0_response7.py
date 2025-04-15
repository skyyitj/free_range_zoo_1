from scipy.spatial import distance
import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    # Compute fire-threat score as fire_level * fire_intensity for all fires
    fire_threat_scores = [lvl * inten for lvl, inten in zip(fire_levels, fire_intensities)]

    # Compute the distance from the agent to all fires
    dist_to_fires = [distance.euclidean(agent_pos, fire) for fire in fire_pos]

    # Compute a score for each fire that penalizes distance and fire-threat score
    fire_scores = [threat - 0.1*dist for threat, dist in zip(fire_threat_scores, dist_to_fires)]
    
    # Also consider the agent's fire reduction power and available suppressant volume 
    fire_scores = [score * agent_fire_reduction_power * agent_suppressant_num for score in fire_scores]

    # Choose the fire with the highest score
    chosen_fire = np.argmax(fire_scores)

    return chosen_fire