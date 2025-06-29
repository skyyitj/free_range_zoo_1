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
    import numpy as np
    
    # Setting temperature parameters for normalization
    distance_temperature = 5.0
    intensity_temperature = 2.0
    reward_temperature = 3.0

    def calculate_distance(agent_pos, fire_pos):
        return np.sqrt((fire_pos[0] - agent_pos[0])**2 + (fire_pos[1] - agent_pos[1])**2)

    num_tasks = len(fire_pos)
    task_scores = []

    for i in range(num_tasks):
        # Calculate distance to fire location
        distance = calculate_distance(agent_pos, fire_pos[i])

        # Distance component: inverse weighting
        distance_score = np.exp(-distance / distance_temperature)

        # Fire intensity component: higher intensity gets higher priority
        intensity_score = np.exp(fire_intensities[i] / intensity_temperature)

        # Fire reward weight component (prioritization based on external weights)
        reward_score = np.exp(fire_putout_weight[i] / reward_temperature)

        # Check available suppressant for this task
        resource_penalty = 0 if agent_suppressant_num > fire_intensities[i] else -1

        # Combine scores (weighted sum)
        combined_score = (distance_score * 0.4) + (intensity_score * 0.4) + (reward_score * 0.2) + resource_penalty
        task_scores.append(combined_score)

    # Select the task with the highest score
    best_task_index = int(np.argmax(task_scores))
    return best_task_index