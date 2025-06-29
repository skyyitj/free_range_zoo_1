import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    num_tasks = len(fire_pos)
    scores = np.zeros(num_tasks)

    # Distance weight temperature
    distance_temperature = 0.05
    importance_temperature = 0.1
    
    for i in range(num_tasks):
        # Calculate the distance from the agent to each fire task
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        distance_score = np.exp(-distance_temperature * distance)
        
        # Calculate the importance of the fire
        # Importance is based on the current intensity and the priority weight
        importance_score = fire_putout_weight[i] * (fire_levels[i] * fire_intensities[i])
        importance_score = np.exp(importance_temperature * importance_score)
        
        # Calculate a composite score
        score = distance_score * importance_score
        scores[i] = score
        
    # Choose the index with the maximum score
    chosen_task = np.argmax(scores)
    return chosen_task