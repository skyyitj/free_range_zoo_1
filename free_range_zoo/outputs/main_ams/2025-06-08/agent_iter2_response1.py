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

    # Define temperature parameters for components
    distance_temperature = 8  # Prioritize proximity
    intensity_temperature = 5  # Strengthen the effect of fire intensity
    suppressant_temperature = 2  # Reward efficient suppressant usage
    burnout_risk_temperature = 4  # Penalize fires close to burning out

    scores = []
    for i in range(len(fire_pos)):
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        
        # Compute distance between agent and fire
        distance = math.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = math.exp(-distance / distance_temperature)

        # Normalize fire intensity as a task priority component
        intensity_score = math.exp(fire_intensities[i] / intensity_temperature)

        # Compute suppressant score
        required_suppressant = fire_intensities[i] / agent_fire_reduction_power
        suppressant_score = (
            math.exp(-required_suppressant / suppressant_temperature)
            if agent_suppressant_num >= required_suppressant
            else 0
        )

        # Add a penalty for fires that are dangerously close to burning out
        burnout_risk = math.exp(fire_levels[i] / burnout_risk_temperature)

        # Final task score combines all components
        total_score = (
            fire_putout_weight[i] * intensity_score * suppressant_score * distance_score / burnout_risk
        )
        scores.append(total_score)

    # Choose the task with the highest score
    return int(scores.index(max(scores)))