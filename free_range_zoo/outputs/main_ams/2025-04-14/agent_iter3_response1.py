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

    # If the agent has low suppressant left, recharge.
    if agent_suppressant_num <= 0.1:  
        return -1

    valid_fires = [i for i, fl in enumerate(fire_levels) if fl < fire_intensities[i]]  # Tasks that aren't self-extinguishing

    if not valid_fires:  
        return -1

    # Calculate scores for each fire. Higher score means higher priority. 
    fire_scores = []
    for i in valid_fires:
        distance = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        score = agent_suppressant_num * fire_intensities[i] / (1 + distance)
        fire_scores.append((score, i))

    # Choose the fire with the highest score.
    task_to_address = max(fire_scores)[1]

    # If there is another agent closer to the fire, let it handle it.
    for other_agent_pos in other_agents_pos:
        if np.hypot(*(np.array(agent_pos) - np.array(other_agent_pos))) < np.hypot(*(np.array(agent_pos) - np.array(fire_pos[task_to_address]))):
            return -1 # let the closer agent handle this fire, this agent can select another fire or wait (-1)

    return task_to_address