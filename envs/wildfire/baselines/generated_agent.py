# Auto-generated Agent Class
from typing import List, Dict, Any
import torch
import free_range_rust
from free_range_zoo.utils.agent import Agent
from math import sqrt
from typing import Tuple, List


class GenerateAgent(Agent):
    """Agent that always fights the strongest available fire."""

    def __init__(self, agent_name: str, parallel_envs: int, **kwargs) -> None:
        """
        Initialize the agent.

        Args:
            agent_name: str - The name of the agent
            parallel_envs: int - The number of parallel environments
            **kwargs: Other parameters, including environment configuration
        """
        super().__init__(agent_name, parallel_envs)
        self.actions = torch.zeros((parallel_envs, 2), dtype=torch.int32)

        # 从配置中获取场地尺寸
        if 'configuration' in kwargs:
            config = kwargs['configuration']
            self.grid_width = config.grid_width
            self.grid_height = config.grid_height
        else:
            # 如果没有配置，从kwargs直接获取
            self.grid_width = kwargs.get('grid_width', 3)
            self.grid_height = kwargs.get('grid_height', 3)
        print(self.grid_width)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        has_suppressant = self.observation['self'][:, 3] != 0
        print(f"self.actions shape: {self.actions.shape}")
        print(f"self.observation['self'] shape: {self.observation['self'].shape}")
        print(f"has_suppressant shape: {has_suppressant.shape}")

        for batch in range(self.parallel_envs):
            maximum_index = -1

            if len(self.argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)  # There are no fires in the environment so agents have to noop
                continue

            agent_obs = self.observation['self'][batch].cpu().numpy()
            agent_pos = agent_obs[0:2].tolist()
            agent_fire_power = float(agent_obs[2])
            agent_suppressant_num = int(agent_obs[3])

            fire_obs = self.observation['tasks'][batch].cpu().numpy()
            fire_pos = fire_obs[:, 0:2].tolist()
            fire_levels = fire_obs[:, 2]
            fire_intensities = fire_obs[:, 3]

            other_agents_obs = self.observation['others'][batch].cpu().numpy()
            other_agents_pos = other_agents_obs.tolist()

            print(f"grid_width: {self.grid_width}, grid_height: {self.grid_height}")
            valid_action_space = torch.full((self.grid_width, self.grid_height), -1, dtype=torch.int32)
            print(f"valid_action_space shape: {valid_action_space.shape}")

            print(f"argmax_store[batch] shape: {self.argmax_store[batch].shape}")
            for i in range(self.argmax_store[batch].size(0)):
                # 获取火点位置
                fire_y = self.argmax_store[batch][i][0]
                fire_x = self.argmax_store[batch][i][1]

                print(f"Fire point {i}: x={fire_x}, y={fire_y}")

                # 确保坐标在有效范围内
                if 0 <= fire_x < self.grid_width and 0 <= fire_y < self.grid_height:
                    if [fire_x, fire_y] in action_space.spaces[0].enumerate():
                        # 将火点的索引存储到valid_action_space中
                        valid_action_space[fire_x][fire_y] = i
                else:
                    print(f"Warning: Fire point {i} out of bounds: x={fire_x}, y={fire_y}")
            maximum_index = single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos,
                                                fire_pos, fire_levels, fire_intensities, valid_action_space)
            print(f"maximum_index: {maximum_index}")

            if maximum_index == -1:  # 若maximum_index没有更新，则说明在action_space中没有可扑灭的火焰
                self.actions[batch].fill_(-1)
            else:
                self.actions[batch, 0] = maximum_index
                self.actions[batch, 1] = 0
                print(f"self.actions[batch] after update: {self.actions[batch]}")

        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop
        print(f"self.actions final shape: {self.actions.shape}")
        print(f"self.actions final content: {self.actions}")

        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        try:
            self.observation, self.t_mapping = observation
            self.t_mapping = self.t_mapping['agent_action_mapping']
        except Exception as e:  # Catch all exceptions
            self.observation = observation

        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]
        print(f"fires shape: {self.fires.shape}")
        self.argmax_store = torch.zeros_like(self.fires)
        print(f"argmax_store shape: {self.argmax_store.shape}")
        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]


def single_agent_policy(
        agent_pos: Tuple[float, float],
        agent_fire_reduction_power: float,
        agent_suppressant_num: float,
        other_agents_pos: List[Tuple[float, float]],
        fire_pos: List[Tuple[float, float]],
        fire_levels: List[float],
        fire_intensities: List[float],
        valid_action_space: List[List[int]]
) -> int:
    """
    Determines the best action for an agent in the wildfire environment.

    Args:
        agent_pos: Position of this agent (y, x)
        agent_fire_reduction_power: Fire suppression power of this agent
        agent_suppressant_num: Number of suppressant available
        other_agents_pos: Positions of all other agents [(y1, x1), (y2, x2), ...] shape: (num_agents-1, 2)
        fire_pos: Positions of all fire tasks [(y1, x1), (y2, x2), ...] shape: (num_tasks, 2)
        fire_levels: Current fire level of each task shape: (num_tasks,)
        fire_intensities: Intensity (difficulty) of each task shape: (num_tasks,)
        valid_action_space: List[List[int]]
            A 2D list representing the environment's grid, where each element indicates the validity of an action at that position.
            -1 indicates no fire (invalid action), while non-negative integers represent valid actions with corresponding fire indices in task.

    Returns:
        int: Index of the chosen task to address (0 to num_tasks-1)
    """
    import random

    # 收集所有有效的动作索引
    valid_indices = []
    for x in range(len(valid_action_space)):
        for y in range(len(valid_action_space[0])):
            if valid_action_space[x][y] != -1:
                valid_indices.append(valid_action_space[x][y])

    # 如果没有有效的动作，返回-1
    if not valid_indices:
        return -1

    # 随机选择一个有效的动作索引
    return random.choice(valid_indices)

