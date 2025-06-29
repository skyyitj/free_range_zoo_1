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
    best_task_score = float('-inf')
    selected_task_index = -1

    # Adjusting model parameters for more effective decision-making
    distance_threshold = 10.0 # a reasonable average distance threshold agent might be expected to cover
    num_agents = len(other_agents_pos) + 1

    for i in range(num_tasks):
        # Calculate the distance between the agent and the fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        
        # Calculate a normalized distance factor
        if distance > distance_threshold:
            distance_factor = 0.5 # penalize long distances
        else:
            distance_factor = 1.0 - (distance / (2 * distance_threshold))
        
        # Consider the current intensity of the fire
        intensity_factor = fire_levels[i] * fire_intensities[i]

        # Effort and impact of suppression
        suppressant_factor = min(agent_suppressant_num, agent_fire_reduction_power / max(1.0, intensity_factor))
        
        # Reward factors
        reward_factor = fire_putout_weight[i]
        
        # Calculate the overall score for taking on this task
        task_score = reward_factor * suppressant_factor * distance_factor

        # Compare score to find the best task
        if task_score > best_task_score:
            best_task_score = task_score
            selected_task_index = i
            
    return selected_task_index