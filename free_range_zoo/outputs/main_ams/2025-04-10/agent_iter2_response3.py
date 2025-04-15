def single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, fire_intensity):
    """
    Policy function for the agent, refined to optimize rewards, suppressant efficiency, and fire intensity reduction.
    """
    
    # Default policy parameters
    SUPPRESSANT_USAGE_EFFICIENCY = 0.7  # Target suppressant efficiency (tuneable)
    MIN_FIRE_INTENSITY_DECREASE = -0.1  # Minimum fire intensity decrease rate
    MAX_FIRE_INTENSITY_DECREASE = -2.0  # Maximum fire intensity decrease rate
    SUPPRESSANT_USAGE_THRESHOLD = 0.5  # Threshold for when to deploy suppressant
    
    # Decision-making logic:
    # The agent's goal is to balance suppressant usage with fire intensity reduction.
    
    # Adjust the fire intensity change based on the agent's position and fire power
    fire_intensity_change = max(MIN_FIRE_INTENSITY_DECREASE, min(MAX_FIRE_INTENSITY_DECREASE, -fire_intensity / (1 + agent_fire_power)))
    
    # Calculate suppressant usage based on current fire intensity and available suppressant
    if fire_intensity > 1.0:  # High fire intensity, use more suppressant
        suppressant_usage = min(agent_suppressant_num, 2.0)  # Use more suppressant if available
    elif fire_intensity > 0.5:  # Medium fire intensity, moderate suppressant usage
        suppressant_usage = min(agent_suppressant_num, 1.0)  # Use less suppressant but still effective
    else:  # Low fire intensity, conserve suppressant
        suppressant_usage = 0.0  # Don't use suppressant unless fire intensity is high
    
    # Recalculate suppressant efficiency based on usage
    suppressant_efficiency = fire_intensity_change / suppressant_usage if suppressant_usage > 0 else 0.0
    
    # Refine the policy decision to improve stability and reward maximization
    if suppressant_efficiency < SUPPRESSANT_USAGE_EFFICIENCY:
        # Under-usage of suppressant, so adjust behavior to increase suppressant deployment
        suppressant_usage += 0.5  # Increase suppressant usage to improve efficiency
    elif suppressant_efficiency > SUPPRESSANT_USAGE_EFFICIENCY:
        # Over-usage of suppressant, so reduce deployment to conserve resources
        suppressant_usage = max(0, suppressant_usage - 0.5)  # Decrease suppressant usage if inefficient
    
    # Final decision-making logic: Apply the optimized suppressant usage to fire intensity change
    fire_intensity_change = fire_intensity_change * (suppressant_usage / agent_suppressant_num)  # Adjust intensity change based on suppressant usage
    
    # Returning a structured decision output
    return {
        'fire_intensity_change': fire_intensity_change,
        'suppressant_usage': suppressant_usage,
        'suppressant_efficiency': suppressant_efficiency
    }