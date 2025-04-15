def single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, fire_intensity, step):
    # Define constants for resource efficiency and suppression thresholds
    FIRE_INTENSITY_THRESHOLD = 0.3  # Threshold for when to use suppressant efficiently
    SUPPRESSANT_USE_THRESHOLD = 0.2  # Minimum fire intensity to trigger suppressant use

    # Calculate the fire intensity difference from previous step
    fire_intensity_change = fire_intensity - previous_fire_intensity
    previous_fire_intensity = fire_intensity  # Update previous fire intensity for next call

    # Basic strategy: prioritize fire suppression if intensity is above a threshold
    if fire_intensity > FIRE_INTENSITY_THRESHOLD and agent_suppressant_num > 0:
        suppressant_use = min(agent_suppressant_num, fire_intensity * SUPPRESSANT_USE_THRESHOLD)
        suppressant_efficiency = suppressant_use / fire_intensity
        total_rewards = (suppressant_efficiency - fire_intensity_change) * 10  # Reward based on efficiency and change in intensity
    else:
        suppressant_use = 0
        suppressant_efficiency = 0
        total_rewards = fire_intensity_change * -5  # Penalty for fire spread when no suppressant is used
    
    # Adjust the fire intensity change per step
    fire_intensity_change_per_step = max(-2.0, fire_intensity_change * 0.8)  # Ensure intensity reduction is reasonable

    # Return the action based on suppressant use and reward maximization
    action = {
        "suppressant_use": suppressant_use,
        "fire_intensity_change_per_step": fire_intensity_change_per_step,
        "total_rewards": total_rewards,
        "suppressant_efficiency": suppressant_efficiency
    }
    
    return action