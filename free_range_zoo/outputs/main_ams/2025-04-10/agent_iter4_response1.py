from typing import List, Tuple

def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
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
        agent_fire_reduction_power: Fire suppression power of this agent
        agent_suppressant_num: Number of suppressant available

        other_agents_pos: Positions of all other agents [(y1, x1), (y2, x2), ...]
        fire_pos: Positions of all fire tasks [(y1, x1), (y2, x2), ...]
        fire_levels: Current fire level of each task
        fire_intensities: Intensity (difficulty) of each task

    Returns:
        int: Index of the chosen task to address (0 to num_tasks-1) or -1 if no task.
    """
    
    # If no fires are active, return -1 indicating no task to handle
    if not fire_levels:
        return -1
    
    # Filter out tasks where the fire level is 0 (i.e., no fire)
    active_fire_indices = [i for i, level in enumerate(fire_levels) if level > 0]
    
    if not active_fire_indices:
        return -1  # No active fires

    # Choose the fire with the highest intensity
    highest_intensity_index = max(active_fire_indices, key=lambda i: fire_intensities[i])

    # Return the index of the task with the highest intensity
    return highest_intensity_index