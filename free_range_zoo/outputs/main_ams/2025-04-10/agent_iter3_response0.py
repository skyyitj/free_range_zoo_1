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

        other_agents_pos: Positions of all other agents [(y1, x1), (y2, x2), ...] shape: (num_agents-1, 2)

        fire_pos: Positions of all fire tasks [(y1, x1), (y2, x2), ...] shape: (num_tasks, 2)
        fire_levels: Current fire level of each task shape: (num_tasks,)
        fire_intensities: Intensity (difficulty) of each task shape: (num_tasks,)

    Returns:
        int: Index of the chosen task to address (0 to num_tasks-1)
    """
    
    # Prioritize fire based on intensity and level (could add more sophisticated logic)
    highest_intensity_index = max(range(len(fire_intensities)), key=lambda i: fire_intensities[i])
    
    # Check if any fire has a high enough level to naturally extinguish (not to be addressed by an agent)
    if fire_levels[highest_intensity_index] >= 100.0:  # Assuming level 100 is the threshold to extinguish automatically
        return -1  # Ignore this task if it will go out naturally
    
    # If the agent has enough suppressant, take action on the most intense fire
    if agent_suppressant_num > 0:
        return highest_intensity_index
    
    # If the agent doesn't have suppressant, it could either recharge or wait for a chance to assist
    return -1  # Return -1 if no action is possible