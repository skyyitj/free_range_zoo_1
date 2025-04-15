from typing import List, Tuple
import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float]
) -> int:

    if agent_suppressant_num <= 0: 
        return -1

    # Calculate distances from the agent to all fires.
    distances = [((fx-agent_pos[0])**2 + (fy-agent_pos[1])**2)**0.5 for fx, fy in fire_pos]
    
    # Also calculate the distance from other agents to fires
    other_agent_distances = [[((fx-x)**2 + (fy-y)**2)**0.5 for fx, fy in fire_pos] for x, y in other_agents_pos]
    min_other_agent_distances = np.min(other_agent_distances, axis=0)

    # Filter out fires that are closer to other agents than to this agent
    fires_to_consider = [i for i, d in enumerate(distances) if d <= min_other_agent_distances[i]]

    if not fires_to_consider:
        return -1

    # Filter out fires that are not threatening to become more intense
    fires_to_consider = [i for i in fires_to_consider if fire_levels[i] > fire_intensities[i]]
    
    if not fires_to_consider:
        return -1

    # Pick the fire with highest level
    return max(fires_to_consider, key=lambda x: fire_levels[x])