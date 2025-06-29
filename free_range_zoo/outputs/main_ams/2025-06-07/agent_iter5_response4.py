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
    import numpy as np

    num_tasks = len(fire_pos)
    
    # Calculate distances from agent to each fire task
    distances = np.array([np.sqrt((fp[0] - agent_pos[0])**2 + (fp[1] - agent_pos[1])**2) for fp in fire_pos])
    
    # Compute efficiencies, such as firepower (how much intensity can an agent reduce)
    firepower = agent_fire_reduction_power * agent_supressant_num
    suppressant_efficiency = firepower / fire_intensities    # effectiveness of agent's current firepower
    
    # Calculate urgency, prioritizing the fires with higher levels but also consider balance
    urgency = (np.array(fire_levels) / max(fire_levels)) * fire_putout_weight
    
    # Improved prioritization: combination of how critical a fire is and how efficiently it can be handled
    scores = suppressant_efficiency * urgency
    
    # Normalize distances; smaller distances should lead to higher scores
    normalized_distances = 1 / (distances + 0.001)
    
    # Modify the total score by adding the effect of normalized distances on it.
    adjusted_scores = scores * normalized_distances
    
    # Return the index of the fire assignment with maximum score
    return int(np.argmax(adjusted_scores))