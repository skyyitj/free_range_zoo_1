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

    # === Adjusting Scoring Criteria ===
    max_score = -float('inf')
    best_fire = None

    dist_temperature = 0.05  # Adjust temperature coefficient to balance distance factor
    suppress_power_temperature = 0.2  # Adjust temperature coefficient to balance suppression power factor

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance factor with higher weight
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Firefighting efficiency factor with lower weight
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity+1)

        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        score = np.exp((fire_weight * suppression_power / (dist_temperature * dist + 1) + fire_level) / suppress_power_temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire

# The above policy calculates a score based on distance to the fire, the agent's suppression power, the intensity of the fire, and its putout weight.
# The function first calculates the Euclidean distance from the agent to the fire. Then, it calculates the suppression power by multiplying 
# the agent's reduction power and its suppressant resources and dividing by one plus the fire intensity. This means that an agent with more 
# resources will have a higher score, but a high-intensity fire will decrease an agent's score. The function then computes the score by applying an exponential transformation
# to the sum of the fire weight multiplied by the suppression power, the fire level, and the distance, all divided by a temperature coefficient.
# This temperature coefficient is used to manually adjust the contributions of the distance and the suppression power to the score. 
# In the end, the function returns the index of the fire with the highest score.
# By returning the index of the fire with the maximum score, the function is assigning the agent to the fire that it'll be able to put out most effectively.