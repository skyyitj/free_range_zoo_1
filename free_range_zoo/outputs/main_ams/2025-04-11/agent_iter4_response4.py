import numpy as np

def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities):
  
    # Calculate distances to all fires, and pair them with their corresponding fire levels and intensities
    fires_data = list(zip(fire_pos, 
                          fire_levels, 
                          fire_intensities, 
                          [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]))
    
    # Check if agent is out of suppressant
    if agent_suppressant_num == 0:
        return None  # agent can't act without suppressant
    
    # Sort the fires data based on a score calculated considering the distance to fire, 
    # the fire level, fire intensity and the agent's fire reduction capability.
    fires_data.sort(key=lambda x: (x[1]/(agent_fire_reduction_power+0.01) + x[2])/(1.0 / (x[3]+0.01)))
    
    # Get the index of the fire with the highest priority 
    highest_priority_fire = fire_pos.index(fires_data[0][0])
    
    return highest_priority_fire