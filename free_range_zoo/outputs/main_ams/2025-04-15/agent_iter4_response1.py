from scipy.spatial import distance
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

    agent_suppressant_capacity = agent_suppressant_num * agent_fire_reduction_power
    
    scores = []
    for i, fp in enumerate(fire_pos):
        # calculate the distance to each fire
        dist = distance.euclidean(agent_pos, fp)
        # calculate potential impact, i.e. compare agent's suppressant capacity vs. fire intensity
        impact = min(agent_suppressant_capacity, fire_intensities[i])
        
        # define weights for different factors in the decision. 
        # Here we emphasize the ability of the agent to control the fire and the distance to the fire.
        dist_weight = 1.0
        impact_weight = 1.5
        level_weight = 0.8
        intensity_weight = 0.8
        weight_weight = 1.0

        # normalized score for each fire task
        score = - dist_weight*np.log(dist+1) \
                - level_weight*np.log(fire_levels[i]+1) \
                - intensity_weight*np.log(fire_intensities[i]+1) \
                + weight_weight*fire_putout_weight[i] \
                + impact_weight*np.log(impact+1)
        scores.append(score)
        
    # choose the fire with the highest score
    return np.argmax(scores)