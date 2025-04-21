In the current policy function, fire with the highest score based on distance, fire intensity, and agent's reduction power is chosen. While this approach takes into consideration the fire intensity and agent's capability, it could still be improved by considering the other agents' positions. 

In situations where multiple agents are close to a severe fire, it could be beneficial to direct more than one agent to the same fire location for faster control, especially if the fire is spreading. This could help prevent the fire from reaching a critical self-extinguishing intensity which would result in penalties.

Let's also add a bias towards preferring fires with higher fire levels as these are more urgent targets. 

By considering these factors into our policy function, we can hopefully improve the performance and efficiency of our fire-fighting agents. 

Let's modify the policy function to reflect these changes:

We add more bias to choose the fire has more levels and which are closer to the agent, simultaneously using the temperature scale(temperature_level and temperature_dist) for the adjustment.

Also, we add factor to consider the other agents' positions to avoid all agents go to fight one fire. It is better to cover a wider range to optimize the overall fire-fighting efficiency. Use temperature_scale to adjust the factor.