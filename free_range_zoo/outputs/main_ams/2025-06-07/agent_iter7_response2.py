def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_supressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
):
    # Standard imports for math operations and manipulation
    import numpy as np

    # Number of firefighting tasks (fires to consider)
    num_fires = len(fire_pos)
    
    # Calculate the distance to each fire from the agent's current position
    distances = [
        np.sqrt((f_y - agent_pos[0])**2 + (f_x - agent_pos[1])**2) for f_y, f_x in fire_pos
    ]
    distances = np.array(distances)

    # Normalized distances (lower is better, so invert)
    distance_score = 1 / (distances + 0.1)  # Adding a small constant to avoid division by zero
    
    # To balance more aggressive suppression with conservative suppressant use:
    # Estimate potential reduction of each fire intensity if this agent works on it
    potential_reductions = agent_fire_reduction_power * agent_supressant_num / np.array(fire_intensities)
    
    # Remaining intensity after suppression
    remaining_fire_intensities = np.array(fire_levels) - potential_reductions
    
    # Target fires which can be potentially put out or significantly reduced, prioritize higher weights
    effectiveness_score = (np.clip(remaining_fire_intensities, 0, None) == 0).astype(float) * fire_putout_weight
    
    # Urgency based on fire level: higher the fire level, more urgent it is
    urgency_score = np.array(fire_levels) / np.max(fire_levels + 0.1)  # normalized urgency
    
    # Combining all scores into a decision metric:
    score = distance_score * effectiveness_score * urgency_score * fire_putout_weight
    
    # Choose the fire task with the highest score
    chosen_fire_index = np.argmax(score)
    
    return chosen_fire_index