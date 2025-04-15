import math

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

    # Criteria to rank fires: (1) Fire level, (2) Intensity, (3) Proximity to this agent, (4) Avoid overlapping efforts
    # Initialize the weights for criteria
    weight_level = 1.0
    weight_intensity = 2.0
    weight_distance = 0.5
    weight_other_agents_proximity = -0.5
    
    best_fire_index = -1
    best_fire_score = float('-inf')

    for i, (pos, level, intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        distance_to_fire = euclidean_distance(agent_pos, pos)
        
        # Count other agents closer to this fire than this agent
        closer_agents_count = sum(1 for other_pos in other_agents_pos if euclidean_distance(other_pos, pos) < distance_to_fire)
        
        # Calculate score for this fire
        score = (weight_level * level +
                 weight_intensity * intensity +
                 weight_distance / (distance_to_fire + 1) +  # avoid division by zero
                 weight_other_agents_proximity * closer_agents_count)
        
        # Choose the fire task which has the maximum score
        if score > best_fire_score:
            best_fire_score = score
            best_fire_index = i
     
    return best_fire_index