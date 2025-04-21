def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float, 

    other_agents_pos: List[Tuple[float, float]], 

    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float], 

    fire_putout_weight: List[float]
) -> int:
    # === Fire Task Selection Scoring ===
    max_score = float('-inf')
    best_fire = None
    
    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # Calculate euclidean distance from agent to each fire
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5

        # Calculate the resource consumption needed for the task
        required_suppressant = fire_intensity / agent_fire_reduction_power

        # Determine if the agent has enough resources to handle this fire
        if agent_suppressant_num >= required_suppressant:
            # Score = 2 x Weight × Suppression Potential - Distance × Fire Intensity
            # Increase the priority of the selected task based on the fire weight and fire suppression potential,
            # and decrease the priority based on the distance and current fire intensity.
            score = 2*fire_weight * (agent_suppressant_num * agent_fire_reduction_power/fire_intensity) - dist * fire_intensity
            if score > max_score:
                max_score = score
                best_fire = i
        else:
            score = fire_weight * (agent_suppressant_num * agent_fire_reduction_power/fire_intensity) - dist * fire_intensity
            if score > max_score:
                max_score = score
                best_fire = i
    return best_fire
# This policy function rewards agents who can successfully suppress the fire and puts more emphasis on the task's rewarding weight.
# It discourages assignments that involve traveling far and facing extremely intense fires with insufficient suppressing resources.
# By giving higher consideration to the fire weight and agent's suppressing capability, this policy function could theoretically improve the efficiency of suppressing fires and save more resources.