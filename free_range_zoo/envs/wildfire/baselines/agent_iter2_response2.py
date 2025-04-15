import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    task_scores = []
    epsilon = 1e-8
    for i in range(len(fire_pos)):
        task_pos = fire_pos[i]
        task_fire_level = fire_levels[i]
        task_fire_intensity = fire_intensities[i]
        distance = ((agent_pos[0] - task_pos[0]) ** 2 + (agent_pos[1] - task_pos[1]) ** 2) ** 0.5
        closer_agents = 0
        for other_agent_pos in other_agents_pos:
            other_agent_distance = ((other_agent_pos[0] - task_pos[0]) ** 2 + (other_agent_pos[1] - task_pos[1]) ** 2) ** 0.5
            if other_agent_distance < distance:
                closer_agents += 1
        else:
            task_score = ((task_fire_level - 0.6 * distance * (task_fire_intensity - 0.3 * closer_agents)) * (agent_fire_reduction_power * np.sqrt(agent_suppressant_num + epsilon)))
            task_scores.append((i, task_score))
    if not task_scores:
        return -1  # Caution: This may happen if there are no fire tasks or all fire tasks have other agents closer to them. Optional: To avoid agents becoming idle in these scenarios, one could remove the pass statement, making them fight fire tasks regardless of whether there are other agents closer to the fire.
    else:
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return task_scores[0][0]