Let's modify the policy function by prioritizing fires that are closer to the agent, have higher intensity, and have a higher pay-off. We'll also make the agent preserve suppressant if it's running low on resources.

First, note that our Average Fire Intensity Change has a negative value, this is good as it means that overall we're reducing the fire intensity. But, It's crucial we minimize the number of fires that burnout (Burnedout Number). This means our agents aren't putting out fires quick enough. 

The agents might be choosing to fight fires that are difficult to put out completely or not as pressing in terms of proximity or intensity level. As such, we should prioritize fires that can be both extinguished quickly and are closer to the agent's location. 

This is how we can modify our policy function:
  - We use the "can_put_out_fire" value of an agent to determine whether the agent has enough suppressant to completely put out a fire at a given location. 
  - We also take the fire's distance into account when computing the score; closer fires are preferred.
  - We consider the payoff for putting out a fire by using the fire_putout_weight in our score calculation. 

Finally, having a better resource management could greatly improve the efficiency of the agents. For instance, if an agent doesn't have enough suppressant to put out a fire, it would be more efficient to move to a closer fire, or to a less intense one.

Here's the updated policy function: