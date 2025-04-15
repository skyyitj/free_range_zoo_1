def single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, max_suppressant_usage):
    """
    Agent's policy to reduce fire intensity and maximize rewards by balancing suppressant usage.
    
    Args:
    - agent_pos: Position of the agent.
    - agent_fire_power: The fire power the agent has.
    - agent_suppressant_num: The number of suppressant units available.
    - other_agents_pos: Positions of other agents.
    - max_suppressant_usage: Maximum allowable suppressant usage per step.
    
    Returns:
    - action: Action chosen by the agent.
    """
    # Define some action thresholds and a balancing factor for efficiency
    suppressant_usage_threshold = 0.5  # A threshold to avoid overuse of suppressant
    fire_reduction_target = -0.15  # Desired target for fire intensity change per step
    
    # Compute fire intensity based on the agent's fire power and surrounding agents
    fire_intensity_change = -agent_fire_power / (1 + len(other_agents_pos))  # Example calculation
    
    # Adjust fire intensity change to be within desired thresholds
    if fire_intensity_change < fire_reduction_target:
        action = "Extinguish"  # Action to reduce fire intensity
        action_strength = min(agent_suppressant_num, max_suppressant_usage)  # Avoid using too much suppressant
    else:
        action = "Move"  # Action to move towards the most dangerous fire areas
        action_strength = 0  # No suppressant used for moving
    
    # Optimize suppressant usage based on available suppressant
    if agent_suppressant_num < suppressant_usage_threshold:
        action_strength = 0  # If too little suppressant, avoid using it
    
    # Return the action and strength
    return action, action_strength