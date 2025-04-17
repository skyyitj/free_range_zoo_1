class StrongestBaseline(Agent):
    """Agent that always fights the strongest available fire."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['agent_action_mapping']

        has_suppressant = self.observation['self'][:, 3] != 0
        fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, 3]

        argmax_store = torch.empty_like(self.t_mapping)

        for batch in range(self.parallel_envs):
            for element in range(self.t_mapping[batch].size(0)):
                argmax_store[batch][element] = fires[batch][element]

            if len(argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)  # There are no fires in the environment so agents have to noop
                continue

            self.actions[batch, 0] = argmax_store[batch].argmax(dim=0)
            self.actions[batch, 1] = 0

        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop

class WeakestBaseline(Agent):
    """Agent that always fights the weakest available fire."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['agent_action_mapping']

        has_suppressant = self.observation['self'][:, 3] != 0
        fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, 3]

        argmax_store = torch.empty_like(self.t_mapping)

        for batch in range(self.parallel_envs):
            for element in range(self.t_mapping[batch].size(0)):
                argmax_store[batch][element] = fires[batch][element]

            if len(argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)
                continue

            self.actions[batch, 0] = argmax_store[batch].argmin(dim=0)
            self.actions[batch, 1] = 0

        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop
class ScoreBaseline(Agent):
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
        self_x = self.observation['self'][0][1]
        self_y = self.observation['self'][0][0]
        for batch in range(self.parallel_envs):
            maximum_score = -100
            maximum_index = -1
            if len(self.argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)  # There are no fires in the environment so agents have to noop
                continue
            for index in range(self.argmax_store[batch].size(0)):
                score = 0
                if [index, 0] in action_space.spaces[0].enumerate():  # 确保选取的动作在该Agent的action_space中
                    fire_x = self.argmax_store[batch][index][1]
                    fire_y = self.argmax_store[batch][index][0]
                    distance = torch.sqrt(abs(fire_x - self_x) ** 2 + abs(fire_y - self_y) ** 2)
                    # distance = abs(fire_x-self_x) + abs(fire_y-self_y)
                    intensity = self.argmax_store[batch][index][2]
                    suppresant = self.observation['self'][batch][3]
                    beta = 0.1  # intensity的权重
                    alpha = 1  # distance的权重
                    score = (2 / ((alpha * distance + 1) * (beta * intensity + 4)))  # 函数 h
                    if score > maximum_score:
                        maximum_score = score
                        maximum_index = index

            if maximum_index == -1:  # 若maximum_index没有更新，则说明在action_space中没有可扑灭的火焰
                self.actions[batch].fill_(-1)
            self.actions[batch, 0] = maximum_index
            self.actions[batch, 1] = 0
        print("Generate")
        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop
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
        except Exception as e:  # 捕获所有异常
            self.observation = observation
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]
        # print(self.fires)
        self.argmax_store = torch.zeros_like(self.fires)

        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]