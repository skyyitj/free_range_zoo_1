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
    
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]
    priorities = []
    for i in range(len(fire_pos)):
        if fire_levels[i] > agent_fire_reduction_power:
            priorities.append(fire_intensities[i]*agent_suppressant_num/distances[i])
        else:
            priorities.append(0)

    maximum_priority_index = np.argmax(priorities)
    
    return maximum_priority_index