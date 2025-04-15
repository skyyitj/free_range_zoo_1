from typing import Tuple, List
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
    
    # The agent should not expend its suppressant if it has already been completely used up
    if agent_suppressant_num <= 0:
        return -1  # agent stay still and do nothing

    # Calculate the distances from the agent to all fires.
    distances = np.sqrt([(agent_pos[0]-fp[0])**2 + (agent_pos[1]-fp[1])**2 for fp in fire_pos])

    # Compute how much of the fire level the agent can reduce for each fire.
    potential_reductions = [min(agent_fire_reduction_power * agent_suppressant_num, fl) for fl in fire_levels]

    scores = []
    
    # For each fire compute a score that represents the potential gain in firefighting this fire.
    for idx, (fire_level, fire_intensity, distance, reduction) in enumerate(zip(fire_levels, fire_intensities, distances, potential_reductions)):
        
        # The agent will choose the fire which it could extinguish completely and the one which is closest to it.
        if reduction >= fire_level:  # if the agent can put out the fire completely
            score = fire_intensity / distance  # we prefer to put out intense fires that are nearby
        else:  # if the agent can't put out the fire completely
            score = reduction / (fire_intensity * distance)  # preference is given to less intense nearby fires that the agent can reduce significantly  
        
        scores.append(score)
    
    # the agent chooses the fire with the highest score
    return np.argmax(scores)