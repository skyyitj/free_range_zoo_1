import math
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
    fire_intensities: List[float],
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

    def euclidean_distance(start, end):
        return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    # Updated policy components with new strategy weights
    weight_level = 2.0  # Increased importance on fire level
    weight_intensity = 1.5  # Focus on fire intensity but not overemphasized
    weight_distance = -0.5  # Negative distance influence (closer is better)
    weight_suppressant = 1.0  # Consider the available suppressant
    weight_agent_overlap = -0.3  # Prevent agents from overlapping in the same area too much

    best_fire_index = -1
    best_fire_score = float('-inf')

    for i, (pos, level, intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        # Calculate distance from the agent to the fire
        distance_to_fire = euclidean_distance(agent_pos, pos)

        # Calculate the suppressant efficiency (how much suppressant is needed for this fire)
        suppressant_efficiency = intensity / agent_fire_reduction_power

        # Consider the number of other agents already working on this fire (avoid overlap)
        agent_overlap = sum(1 for other_pos in other_agents_pos if euclidean_distance(other_pos, pos) < distance_to_fire + 2.0)  # Allow some tolerance

        # Score each fire task based on the different components
        score = (weight_level * level + 
                 weight_intensity * intensity +
                 weight_distance / (distance_to_fire + 1) +
                 weight_suppressant * suppressant_efficiency -  # Consider how much suppressant is needed
                 weight_agent_overlap * agent_overlap)

        # Choose the fire task with the maximum score
        if score > best_fire_score:
            best_fire_score = score
            best_fire_index = i

    # Return the index of the best fire task to tackle
    return best_fire_index