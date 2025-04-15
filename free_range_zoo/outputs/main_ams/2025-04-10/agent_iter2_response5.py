def single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, fire_intensity):
    """
    A refined policy function to optimize fire intensity reduction and suppressant efficiency.
    """
    
    # Define a threshold for fire intensity that the agent should target
    TARGET_INTENSITY = 0.05  # Desired fire intensity to reach
    MAX_SUPPRESSANT_USAGE = 5  # Maximum allowable suppressant usage per action
    
    # 1. Decide whether to apply suppressant based on the current fire intensity
    if fire_intensity > TARGET_INTENSITY and agent_suppressant_num > 0:
        # Apply suppressant if fire intensity is high and agent has suppressant available
        suppress_action = True
    else:
        suppress_action = False
    
    # 2. Check how much suppressant should be used (avoid wasting too much)
    if suppress_action:
        # Use a portion of suppressant proportional to remaining fire intensity
        suppressant_usage = min(agent_suppressant_num, max(0.1, fire_intensity * 2))  # Use enough to control fire but not too much
    else:
        suppressant_usage = 0
    
    # 3. Calculate fire intensity change based on suppressant usage
    new_fire_intensity = fire_intensity - suppressant_usage * agent_fire_power
    new_fire_intensity = max(new_fire_intensity, 0)  # Fire intensity can't go below 0
    
    # 4. Calculate the reward based on fire intensity change and suppressant efficiency
    if suppress_action:
        efficiency = suppressant_usage / agent_fire_power  # Calculate efficiency of suppressant usage
    else:
        efficiency = 0
    
    reward = -new_fire_intensity * 10 + efficiency * 100  # Reward is inversely proportional to fire intensity, and positively to efficiency
    
    # 5. Return the new fire intensity and the calculated action
    return new_fire_intensity, reward, suppress_action