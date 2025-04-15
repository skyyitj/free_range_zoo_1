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
    
    # determine the distance between agent and fires
    distance_to_fires = [((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2)**0.5 for fire in fire_pos]
    # determine the distance between other agents and fires
    distance_other_agents_to_fires = [[((fire[0]-agent[0])**2 + (fire[1]-agent[1])**2)**0.5 for fire in fire_pos] for agent in other_agents_pos]

    # take selected fire which can be extinguishing in one action by this agent and is within distance
    valid_fire_indices = [idx for idx, intensity in enumerate(fire_intensities) if intensity <= agent_fire_reduction_power and agent_suppressant_num > 0 and distance_to_fires[idx] <= agent_suppressant_num]

    # get fires that are not targeted by other agents or they cannot reach in one action
    untargeted_fire_indices = [idx for idx, fire in enumerate(fire_pos) if all(distance > agent_suppressant_num for distance in distance_other_agents_to_fires[idx])]

    # intersect valid and untargeted fires
    valid_untargeted_indices = list(set(valid_fire_indices) & set(untargeted_fire_indices))

    if valid_untargeted_indices:
        # greedy strategy for valid untargeted indices - choose one with highest intensity
        chosen_fire = max(valid_untargeted_indices, key=lambda idx: fire_intensities[idx])
    else:
        # fallback strategy - choose one from valid fires
        if valid_fire_indices:
            chosen_fire = max((i for i in valid_fire_indices), key=lambda idx: fire_intensities[idx])
        else:
            # worst case - choose nearest fire
            chosen_fire = distance_to_fires.index(min(distance_to_fires))

    return chosen_fire