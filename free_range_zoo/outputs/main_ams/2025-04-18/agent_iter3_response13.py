import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float, 

    other_agents_pos: List[Tuple[float, float]], 

    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float], 

    fire_putout_weight: List[float]) -> int:

    # Compute distance from agent to each fire
    distances = [np.linalg.norm(np.array(agent_pos) - np.array(fp)) for fp in fire_pos]
    
    # Compute the agent's expected fire putout power
    expected_putout_powers = [min(fire_intensities[i], agent_fire_reduction_power*agent_suppressant_num) for i in range(len(fire_pos))]
    
    # Compute score for each fire (higher is better)
    scores = [fire_putout_weight[i] * expected_putout_powers[i] / max(distances[i], 1.0) for i in range(len(fire_pos))]
    
    # Return the index of the fire with the highest score
    return np.argmax(scores)