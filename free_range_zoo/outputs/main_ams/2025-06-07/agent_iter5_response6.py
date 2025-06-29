def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
):
    import numpy as np

    # Calculate distances of the agent to each fire
    distances = np.sqrt((np.array(fire_pos)[:, 0] - agent_pos[0])**2 + (np.array(fire_pos)[:, 1] - agent_pos[1])**2)

    # Calculate potential effects of agent's suppressant use on each fire
    potential_fire_reductions = (agent_suppressant_num * agent_fire_reduction_power) / np.array(fire_intensities)
    remaining_fire = np.array(fire_levels) - potential_fire_reductions
    
    # Identify fires that can be extinguished in one move (optimal targets)
    extinguishable = remaining_fire <= 0
    
    # Calculate how much suppressant would remain after fire intervention (if none is removed, this is just agent_suppressant_num)
    remaining_suppressant = agent_suppressant_num - (np.array(fire_intensities) / agent_fire_reduction_power)
    
    # ----- Evaluation metrics used to prioritize fires:
    
    # Distance factor: prefer closer fires
    inv_distances = 1 / (distances + 1e-5)  # use small epsilon to avoid zero division
    
    # Desperation factor: percentual potential intensity reduction (we want it high)
    intensity_change = 1 - (remaining_fire / np.array(fire_levels, dtype=np.float32))
    
    # Emphasize more on fires that can be extinguished
    suppressant_factor = extinguishable * (remaining_suppressant >= 0)
    
    # Using the reward weight as direct multiplier
    weighted_scores = (inv_distances * intensity_change * suppressant_factor * np.array(fire_putout_weight))
    
    # Choose the fire target with the highest score
    target_fire_index = np.argmax(weighted_scores)

    return target_fire_index