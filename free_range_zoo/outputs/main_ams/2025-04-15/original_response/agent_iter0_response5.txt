import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    # calculate the distance to each fire from the agent
    distances_to_fires = [np.linalg.norm(np.array(agent_pos)-np.array(fire), ord=2) for fire in fire_pos]

    # calculate the distance to each fire from other agents
    distances_to_fires_from_others = [[np.linalg.norm(np.array(agent)-np.array(fire), ord=2) for agent in other_agents_pos] for fire in fire_pos]

    # calculate the minimum distance to each fire from other agents
    min_distances_to_fires_from_others = [min(distances) if distances else np.inf for distances in distances_to_fires_from_others]

    scores = []
    for fire_index, (distance, other_distance, intensity, weight) in enumerate(zip(distances_to_fires, min_distances_to_fires_from_others, fire_intensities, fire_putout_weight)):

        # calculate the number of steps required to put out the fire
        steps_to_put_out = intensity / agent_fire_reduction_power

        # check if the agent has enough suppressant to put out the fire
        if steps_to_put_out > agent_suppressant_num:
            # if not, ignore this fire
            continue

        # calculate the score for this fire
        score = - weight * distance + other_distance - steps_to_put_out

        # append the score and index to the scores list
        scores.append((score, fire_index))

    # sort the scores list in descending order
    scores.sort(reverse=True)

    # return the index of the fire with the highest score
    return scores[0][1] if scores else -1