# Auto-generated Agent Class
from typing import List, Dict, Any
import torch
import free_range_rust
from free_range_zoo.utils.agent import Agent


class StrongestBaseline(Agent):
    """Agent that always fights the strongest available fire."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)
        print("Generate")
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        has_suppressant = self.observation['self'][:, 3] != 0
        # 记录agent坐标
        self_x = self.observation['self'][0][1]
        self_y = self.observation['self'][0][0]

        # print("Action_space", action_space.spaces[0].enumerate())
        # 查看空间对象的属性和方法
        # print(dir(action_space.spaces))

        # for space in action_space.spaces:
        #     if hasattr(space, 'spaces'):
        #         for idx, sub_space in enumerate(space):
        #             print(f"  子空间 {idx}:")
        #             print(f"    类型: {type(sub_space).__name__}")

        #             # 如果是Discrete空间，打印其属性
        #             if hasattr(sub_space, 'n') and hasattr(sub_space, 'start'):
        #                 print(f"    n值: {sub_space.n}")
        #                 print(f"    起始值: {sub_space.start}")

        for batch in range(self.parallel_envs):
            maximum_score = -100
            # print(argmax_store[batch])
            # sorted_values, sorted_indices = torch.sort(self.argmax_store[batch])
            maximum_index = -1
            # print("fire count: ", self.argmax_store[batch].size(0))

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


            maximum_index = single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, fire_pos, fire_levels, fire_intensities)

            if maximum_index == -1:  # 若maximum_index没有更新，则说明在action_space中没有可扑灭的火焰
                self.actions[batch].fill_(-1)

            self.actions[batch, 0] = maximum_index
            self.actions[batch, 1] = 0
        print("Generate")
        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop

        # print(self.actions)

        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        # print(observation)

        # 跑main.py和evaluate.py时的api有所不同，因此采取不同的处理方式
        try:
            self.observation, self.t_mapping = observation
            self.t_mapping = self.t_mapping['agent_action_mapping']
        except Exception as e:  # 捕获所有异常
            self.observation = observation


        
        # self.observation = observation

        # self.observation, self.t_mapping = observation
        # self.t_mapping = self.t_mapping['agent_action_mapping']

        # 记录火焰的位置与强度

        # self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]
        # # print(self.fires)
        # self.argmax_store = torch.zeros_like(self.fires)

        # for batch in range(self.parallel_envs):
        #     for element in range(self.fires[batch].size(0)):
        #         self.argmax_store[batch][element] = self.fires[batch][element]