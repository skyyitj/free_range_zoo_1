def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], 

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                      
    fire_intensities: List[float],               

    # === Task Prioritization ===
    fire_putout_weight: List[float],             
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    import math

    # Temperature parameters for balancing various factors
    distance_temperature = 7    # Proximity factor
    intensity_temperature = 8   # Fire severity scaling
    suppressant_temperature = 1 # Resource efficiency
    level_penalty_temperature = 3  # Penalize critical fire levels

    scores = []
    for i in range(len(fire_pos)):
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        
        # Calculate distance between agent and fire
        distance = math.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = math.exp(-distance / distance_temperature)
        
        # Fire severity is prioritized by its intensity
        intensity_score = math.exp(fire_intensities[i] / intensity_temperature)
        
        # Calculate required suppressant based on fire intensity and agent power
        required_suppressant = fire_intensities[i] / agent_fire_reduction_power
        suppressant_score = (
            math.exp(-required_suppressant / suppressant_temperature) 
            if agent_suppressant_num >= required_suppressant else 0
        )
        
        # Apply penalty for fires with dangerously high levels
        level_penalty = math.exp(fire_levels[i] / level_penalty_temperature)

        # Adjust final task score by incorporating reward weights
        total_score = (
            fire_putout_weight[i] * suppressant_score * intensity_score * 
            distance_score / level_penalty
        )
        scores.append(total_score)

    # Choose the task with the highest score
    return int(scores.index(max(scores)))