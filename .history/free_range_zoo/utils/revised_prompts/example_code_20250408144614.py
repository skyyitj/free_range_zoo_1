from typing import Tuple, List

def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_power: float,
    agent_suppressant_num: float,
    
    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],
    
    # Task information
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],

    valid_tasks: List[bool]
) -> int:
    """
    Determines the best action for an agent in the wildfire environment.
    
    Args:
        agent_pos: Position of this agent (y, x)
        agent_fire_power: Fire suppression power of this agent
        agent_suppressant_num: Number of suppressant available
        
        other_agents_pos: Positions of all other agents [(y1, x1), (y2, x2), ...] shape: (num_agents-1, 2)
        
        fire_pos: Positions of all fire tasks [(y1, x1), (y2, x2), ...] shape: (num_tasks, 2)
        fire_levels: Current fire level of each task shape: (num_tasks,)
        fire_intensities: Intensity (difficulty) of each task shape: (num_tasks,)

        valid_tasks: List[bool] - Whether each task is valid to be addressed by the agent
    Returns:
        int: Index of the chosen task to address (0 to num_tasks-1)
    """
    num_tasks = len(fire_pos)
    for index in range(num_tasks):
        score = 0
        if valid_tasks[index]:  # 确保选取的动作在该Agent的action_space中

            # 进行h（score）的计算
            fire_x = self.argmax_store[batch][index][1]
            fire_y = self.argmax_store[batch][index][0]
            distance = torch.sqrt(abs(fire_x - self_x) ** 2 + abs(fire_y - self_y) ** 2)
            # distance = abs(fire_x-self_x) + abs(fire_y-self_y)
            intensity = self.argmax_store[batch][index][2]
            suppresant = self.observation['self'][batch][3]
            beta = 0.1  # intensity的权重
            alpha = 1  # distance的权重
            # score = (2 / ((alpha * distance + 0.5) * (beta * intensity + 0.5)) ) + (suppresant - intensity) #函数 h
            score = (2 / ((alpha * distance + 1) * (beta * intensity + 4)))  # 函数 h
            # print("distance:", distance)
            # print("intensity: ", intensity)
            # print("score: ", score)

            if score > maximum_score:
                maximum_score = score
                maximum_index = index
    pass


