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
        int: Index of the chosen task to address (0 to num_tasks-1)
    """
    # The agent chooses the fire to tackle based on some strategy.
    # Here's a simple strategy as an example:
    
    best_fire_idx = -1
    max_fire_intensity = -1
    for i, fire in enumerate(fire_pos):
        if fire_levels[i] > 0:  # If fire is still active
            if fire_intensities[i] > max_fire_intensity:  # Maximize intensity tackled
                best_fire_idx = i
                max_fire_intensity = fire_intensities[i]
    
    return best_fire_idx