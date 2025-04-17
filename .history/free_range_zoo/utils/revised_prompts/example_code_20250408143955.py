def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_power: float,
    agent_suppressant_num: float,
    
    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],
    
    # Task information
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float]
) -> int:
    """
    Determines the best action for an agent in the wildfire environment.
    
    Args:
        agent_pos: Position of this agent (y, x)
        agent_fire_power: Fire suppression power of this agent
        agent_suppressant_num: Number of suppressant available
        
        other_agents_pos: Positions of all other agents [(y1, x1), (y2, x2), ...] shape: (num_agents-1, 2)
        
        fire_pos: Positions of all fire tasks [(y1, x1), (y2, x2), ...] shape: (num_tasks, 2)
        fire_levels: Current fire level of each task shape: (num_tasks,)
        fire_intensities: Intensity (difficulty) of each task shape: (num_tasks,)
    
    Returns:
        int: Index of the chosen task to address (0 to num_tasks-1)
    """
    
    pass


