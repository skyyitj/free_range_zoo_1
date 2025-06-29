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
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Revised version of the policy. The new model considers the fire intensity level in addition
    to the agent's fire reduction power and the priority weight of each fire. Additionally, it
    prioritizes fires that have already assigned the least number of agents.
    """
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    distance_normalization_temp = 0.1
    intensity_normalization_temp = 0.05
    assigned_agents_normalization_temp = 0.1
    
    for i in range(num_tasks):
        # calculate distance between agent and fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        # consider intensity level of the fire
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        
        # calculate how many other agents are assigned to this fire
        num_assigned_agents = sum([np.array_equal(pos, fire_pos[i]) for pos in other_agents_pos])
        norm_assigned_agents = np.exp(assigned_agents_normalization_temp * num_assigned_agents) if num_assigned_agents > 0 else 0.0

        agent_efficiency = agent_fire_reduction_power / (1.0 + intensity)
        suppressant_limitation = 1.0 / (1.0 + agent_suppressant_num)
        
        score = fire_putout_weight[i] * agent_efficiency * norm_distance * norm_intensity * (1 - norm_assigned_agents) * suppressant_limitation 

        # Compare score to find the best task
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index