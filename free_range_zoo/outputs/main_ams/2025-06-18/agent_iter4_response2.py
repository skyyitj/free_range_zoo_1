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
    import numpy as np
    
    # Parameters for transformation temperatures
    distance_temp = 10.0  # Temperature for distance scoring
    intensity_temp = 5.0  # Temperature for intensity scoring
    priority_temp = 15.0  # Temperature for priority scoring
    
    scores = []
    # Iterate over all fire tasks
    for i in range(len(fire_pos)):
        # Extract fire task properties
        fire_location = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        priority_weight = fire_putout_weight[i]
        
        # Calculate distance between agent and fire location
        distance = np.sqrt((fire_location[0] - agent_pos[0])**2 + (fire_location[1] - agent_pos[1])**2)
        
        # Transform distance score (closer fires preferred)
        distance_score = np.exp(-distance / distance_temp)
        
        # Transform fire intensity score (higher intensity fires prioritized)
        intensity_score = np.exp(fire_intensity / intensity_temp)
        
        # Transform priority score based on task weight
        priority_score = np.exp(priority_weight / priority_temp)
        
        # Combine scores into a single score for this fire task
        # Higher scores preferred: closer fires, higher intensity fires, higher priority weights
        combined_score = distance_score * intensity_score * priority_score
        
        # Penalize tasks that might waste resources if fire level is above some threshold
        if fire_level > 10:  # Example of a fire level threshold
            combined_score *= 0.1  # Reduce priority for fires above threshold
            
        # Account for remaining suppressant resources (only commit if agent has enough resources)
        if fire_intensity > (agent_fire_reduction_power * agent_suppressant_num):
            combined_score *= 0.2  # Reduce score if agent can't fully suppress the fire
        
        scores.append(combined_score)
    
    # Select the task with the highest score
    selected_task_index = int(np.argmax(scores))
    return selected_task_index