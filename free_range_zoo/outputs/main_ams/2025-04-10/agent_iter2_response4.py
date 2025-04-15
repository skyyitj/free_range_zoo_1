def single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, environment_state):
    """
    This function defines the policy for a single agent in the wildfire environment.
    The goal is to reduce fire intensity while optimizing suppressant efficiency.

    Args:
    - agent_pos: Current position of the agent.
    - agent_fire_power: The firepower available to the agent.
    - agent_suppressant_num: The amount of suppressant the agent has.
    - other_agents_pos: Positions of other agents.
    - environment_state: The overall state of the environment, including fire intensity, fire spread, etc.

    Returns:
    - action: The chosen action for the agent.
    """
    # Extract the environment state (this could include fire intensity, agent positions, etc.)
    fire_intensity = environment_state['fire_intensity']
    fire_location = environment_state['fire_location']
    
    # Action selection strategy based on fire intensity
    if fire_intensity > 0.5:  # If fire intensity is high, prioritize using suppressant aggressively
        if agent_suppressant_num > 0:
            action = 'apply_suppressant'
            suppressant_efficiency = 1.0  # High efficiency when suppressant is applied
        else:
            action = 'move_toward_fire'  # Move towards the fire to get closer for a potential suppressant use
            suppressant_efficiency = 0.0  # No suppressant available
    elif fire_intensity > 0.2:  # Medium fire intensity, balance between moving and applying suppressant
        if agent_suppressant_num > 1:
            action = 'apply_suppressant'
            suppressant_efficiency = 0.75  # Some efficiency in suppressing the fire
        else:
            action = 'move_toward_fire'
            suppressant_efficiency = 0.5
    else:  # Low fire intensity, no urgent need to apply suppressant, focus on repositioning
        action = 'patrol'  # Move randomly or perform other tasks like repositioning
        suppressant_efficiency = 0.0  # Not actively using suppressant

    # Return the chosen action along with the corresponding suppressant efficiency
    return action, suppressant_efficiency