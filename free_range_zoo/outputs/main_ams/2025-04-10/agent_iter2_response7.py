def single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, fire_intensity):
    """
    Refined policy function for the agent.
    
    Parameters:
    - agent_pos: position of the agent
    - agent_fire_power: available fire power of the agent
    - agent_suppressant_num: amount of suppressant available to the agent
    - other_agents_pos: positions of other agents
    - fire_intensity: current fire intensity in the environment

    Returns:
    - action: the chosen action based on the agent's policy
    """
    
    # Step 1: Define thresholds and efficiency targets
    MAX_SUPPRESSANT_EFFICIENCY = 1.3333
    MIN_FIRE_INTENSITY = 0.1  # Threshold below which the agent should reduce suppressant usage
    
    # Step 2: Define the policy for fire suppression
    if fire_intensity > MIN_FIRE_INTENSITY:
        # Prioritize suppressant usage if the fire intensity is high
        if agent_suppressant_num > 0:
            action = "use_suppressant"
            suppressant_efficiency = min(agent_fire_power / agent_suppressant_num, MAX_SUPPRESSANT_EFFICIENCY)
        else:
            action = "move_closer"  # Move towards the fire to collect more suppressant or assist other agents
            suppressant_efficiency = 0
    else:
        # If the fire is nearly under control, either reduce usage or move to assist other agents
        action = "move_closer"
        suppressant_efficiency = 0
    
    # Step 3: Adjust agent behavior based on agent position and other agents' positions
    # If other agents are closer and performing well, the agent might help them or move to a new area.
    if other_agents_pos and agent_pos != other_agents_pos[0]:
        action = "assist_other_agent"
    
    # Return action and the calculated suppressant efficiency
    return action, suppressant_efficiency