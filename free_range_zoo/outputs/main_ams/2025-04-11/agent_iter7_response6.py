import numpy as np

def single_agent_policy(agent_pos, agent_fire_reduction_power, agent_suppressant_num, other_agents_pos, fire_pos, fire_levels,  fire_intensities):
    
    # If there are no fires or the agent has used up its suppressant, it should recharge
    if not fire_pos or agent_suppressant_num <= 0:
        return -1

    # Calculate distances from the agent to each fire
    distances_to_fire = np.sqrt(np.sum((np.array(fire_pos) - np.array(agent_pos))**2, axis=1))

    # We only consider fires that can be extinguished with the current amount of suppressant
    extinguishable_fires_indices = np.nonzero(distances_to_fire <= agent_suppressant_num)[0]

    if extinguishable_fires_indices.size == 0:
        return -1 

    # Find the amount of suppressant required for each fire
    suppressant_required = np.array(fire_levels) / agent_fire_reduction_power

    # We only consider fires that we can extinguish with our current amount of suppressant
    extinguishable_fire_levels = suppressant_required[extinguishable_fires_indices]

    # Now we give priority to the fires we can extinguish, considering both the fire level and intensity
    # We weigh them equally for now, but this could be tweaked
    priority = extinguishable_fire_levels / np.array(fire_intensities)[extinguishable_fires_indices]

    # Choose the fire with the highest priority
    chosen_fire_index = extinguishable_fires_indices[np.argmax(priority)]
    return chosen_fire_index