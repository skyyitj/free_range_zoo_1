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
    Choose the optimal fire-fighting task for a single agent.
    """
    num_tasks = len(fire_pos)
    best_task_index = 0
    max_score = -np.inf

    # Temperatures for weighted score components
    distance_temp = 0.1
    suppressant_temp = 1.
    
    # Iterate through each fire task and calculate a score
    for i in range(num_tasks):
        # Calculate distance from agent to fire task
        dist = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))
        
        # Reduce fire intensity factoring in agent's fire reduction power and suppressant
        potential_fire_reduction = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensities[i])
        
        # Calculate score components
        distance_score = np.exp(-distance_temp * dist)  # prefer tasks that are closer
        fire_reduction_score = potential_fire_reduction  # prefer tasks where more fire can be reduced
        resource_score = np.exp(-suppressant_temp * (1 - (agent_suppressant_num / np.maximum(potential_fire_reduction, 0.1))))  # prioritize conservation of resources
        task_importance_score = fire_putout_weight[i]  # weighted importance of the task

        # Calculate final score for this task
        score = distance_score * fire_reduction_score * resource_score * task_importance_score
        
        # Update the best task based on score
        if score > max_score:
            max_score = score
            best_task_index = i

    return best_task_index

# The function should now guided by distance, agent's fire reduction potential, resource management and the importance weight of each task.