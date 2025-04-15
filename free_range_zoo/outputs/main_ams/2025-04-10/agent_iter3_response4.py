def act(self, action_space):
    # Example agent state (these values should come from your environment or agent’s current state)
    agent_pos = self.get_position()  # Assuming this method exists
    agent_fire_reduction_power = self.get_fire_reduction_power()  # Assuming this method exists
    agent_suppressant_num = self.get_suppressant_num()  # Assuming this method exists
    
    # Get other agents' states
    other_agents_pos = self.get_other_agents_positions()  # Assuming this method exists

    # Get fire-related information (position, level, intensity)
    fire_pos = self.get_fire_positions()  # Assuming this method exists
    fire_levels = self.get_fire_levels()  # Assuming this method exists
    fire_intensities = self.get_fire_intensities()  # Assuming this method exists

    # Call the policy to decide which fire to tackle
    action = single_agent_policy(
        agent_pos,
        agent_fire_reduction_power,
        agent_suppressant_num,
        other_agents_pos,
        fire_pos,
        fire_levels,
        fire_intensities
    )

    return action