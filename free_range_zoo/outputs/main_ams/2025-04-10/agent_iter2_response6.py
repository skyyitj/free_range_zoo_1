def single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, action_space):
    """
    The policy function that determines the actions of the agent based on its position,
    fire power, and suppressant count, while also considering the actions of other agents.
    
    Parameters:
    - agent_pos: The position of the agent.
    - agent_fire_power: The fire power the agent possesses.
    - agent_suppressant_num: The amount of suppressant the agent has.
    - other_agents_pos: The positions of other agents.
    - action_space: The available actions the agent can take.

    Returns:
    - action: The selected action.
    """

    # Define decision thresholds for suppressant usage
    suppressant_threshold = 0.5  # Minimum efficiency threshold for using suppressant
    fire_intensity_threshold = -0.1  # Fire intensity change per step below which we decide to use more suppressant
    
    # If the agent has suppressant and fire intensity is still high, we prioritize using suppressant.
    if agent_suppressant_num > 0 and agent_fire_power > fire_intensity_threshold:
        if agent_fire_power > suppressant_threshold:
            action = "use_suppressant"  # Use suppressant when fire intensity is still high and suppressant is available
        else:
            action = "move_towards_fire"  # If fire intensity is lower, move towards the fire to reduce it further
    else:
        action = "move_random"  # If no suppressant or fire intensity is too low, move randomly or reposition
    
    return action