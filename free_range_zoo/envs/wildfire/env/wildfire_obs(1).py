"""
# Specification
| Actions            | Discrete & Stochastic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Observations       | Discrete and fully Observed with private observations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Agent Names        | [$firefighter$_0, ..., $firefighter$_n]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| # Agents           | [0, $n_{firefighters}$]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Action Shape       | ($envs$, 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Action Values      | [$fight_0$, ..., $fight_{tasks}$, $noop$ (-1)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Observation Shape  | TensorDict: { <br>&emsp;**self**: $<ypos, xpos, fire power, suppressant>$<br>&emsp;**others**: $<ypos,xpos,fire power, suppressant>$<br>&emsp;**tasks**: $<y, x, fire level, intensity>$ <br> **batch_size**: $num\_envs$ }                                                                                                                                                                                                                                                                                                                                                                      |
| Observation Values | <u>**self**</u>:<br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_power\_reduction$: $[0, max_{fire\_power\_reduction}]$<br>&emsp;$suppressant$: $[0, max_{suppressant}]$<br><u>**others**</u>:<br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_power\_reduction$: $[0, max_{fire\_power\_reduction}]$<br>&emsp;$suppressant$: $[0, max_{suppressant}]$<br> <u>**tasks**</u><br>&emsp;$ypos$: $[0, max_y]$<br>&emsp;$xpos$: $[0, max_x]$<br>&emsp;$fire\_level$: $[0, max_{fire\_level}]$<br>&emsp;$intensity$: $[0, num_{fire\_states}]$ |
"""
def action_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the action space for the given agent.
        Args:
            agent: str - the name of the agent to retrieve the action space for
        Returns:
            List[gymnasium.Space]: the action space for the given agent
        """
        if self.show_bad_actions:
            num_tasks_in_environment = self.environment_task_count
        else:
            num_tasks_in_environment = self.agent_task_count[self.agent_name_mapping[agent]]

        return actions.build_action_space(environment_task_counts=num_tasks_in_environment)
def observation_space(self, agent: str) -> List[gymnasium.Space]:
        """
        Return the observation space for the given agent.

        Args:
            agent: str - the name of the agent to retrieve the observation space for
        Returns:
            List[gymnasium.Space]: the observation space for the given agent
        """
        return observations.build_observation_space(
            environment_task_counts=self.environment_task_count,
            num_agents=self.agent_config.num_agents,
            agent_high=self.agent_observation_bounds,
            fire_high=self.fire_observation_bounds,
            include_suppressant=self.observe_other_suppressant,
            include_power=self.observe_other_power,
        )
def step(self, actions: torch.Tensor) -> None:
    if torch.all(self.terminations[self.agent_selection]) or torch.all(self.truncations[self.agent_selection]):
        return
    self._clear_rewards()
    agent = self.agent_selection
    self.actions[agent] = actions
    # Step the environment after all agents have taken actions
    if self.agent_selector.is_last():
        # Handle the stepping of the environment itself and update the AEC attributes
        rewards, terminations, infos = self.step_environment()
        self.rewards = rewards
        self.terminations = terminations
        self.infos = infos

        # Increment the number of steps taken in each batched environment
        self.num_moves += 1

        if self.max_steps is not None:
            is_truncated = self.num_moves >= self.max_steps
            for agent in self.agents:
                self.truncations[agent] = is_truncated

        self._accumulate_rewards()
        self.update_observations()
        self.update_actions()

        if self.log_directory is not None:
            self._log_environment()

    self.agent_selection = self.agent_selector.next()
def reset(self, seed: Union[List[int], int] = None, options: Dict[str, Any] = None) -> None:
    super().reset(seed=seed, options=options)
    self.agent_action_to_task_mapping = {agent: {} for agent in self.agents}
    self.fire_reduction_power = self.agent_config.fire_reduction_power
    self.suppressant_states = self.agent_config.suppressant_states

    self._state = WildfireState(
        fires=torch.zeros(
            (self.parallel_envs, self.max_y, self.max_x),
            dtype=torch.int32,
            device=self.device,
        ),
        intensity=torch.zeros(
            (self.parallel_envs, self.max_y, self.max_x),
            dtype=torch.int32,
            device=self.device,
        ),
        fuel=torch.zeros(
            (self.parallel_envs, self.max_y, self.max_x),
            dtype=torch.int32,
            device=self.device,
        ),
        agents=self.agent_config.agents,
        capacity=torch.ones(
            (self.parallel_envs, self.agent_config.num_agents),
            dtype=torch.float32,
            device=self.device,
        ),
        suppressants=torch.ones(
            (self.parallel_envs, self.agent_config.num_agents),
            dtype=torch.float32,
            device=self.device,
        ),
        equipment=torch.ones(
            (self.parallel_envs, self.agent_config.num_agents),
            dtype=torch.int32,
            device=self.device,
        ),
    )

    if options is not None and options.get('initial_state') is not None:
        initial_state = options['initial_state']
        if len(initial_state) != self.parallel_envs:
            raise ValueError("Initial state must have the same number of environments as the parallel environments")
        self._state = initial_state
    else:
        self._state.fires[:, self.fire_config.lit] = self.fire_config.fire_types[self.fire_config.lit]
        self._state.fires[:, ~self.fire_config.lit] = -1 * self.fire_config.fire_types[~self.fire_config.lit]
        self._state.intensity[:, self.fire_config.lit] = self.ignition_temp[self.fire_config.lit]
        self._state.fuel[self._state.fires != 0] = self.fire_config.initial_fuel

        self._state.suppressants[:, :] = self.agent_config.initial_suppressant
        self._state.capacity[:, :] = self.agent_config.initial_capacity
        self._state.equipment[:, :] = self.agent_config.initial_equipment_state
    self._state.save_initial()
    # Initialize the rewards for all environments
    self.fire_rewards = self.reward_config.fire_rewards.unsqueeze(0).expand(self.parallel_envs, -1, -1)
    self.num_burnouts = torch.zeros(self.parallel_envs, dtype=torch.int32, device=self.device)
    # Intialize the mapping of the tasks "in" the environment, used to map actions
    self.agent_action_mapping = {agent: None for agent in self.agents}
    self.agent_observation_mapping = {agent: None for agent in self.agents}
    self.agent_bad_actions = {agent: None for agent in self.agents}
    # Set the observations and action space
    if not options or not options.get('skip_observations', False):
        self.update_observations()
    if not options or not options.get('skip_actions', False):
        self.update_actions()
    self._post_reset_hook()
# Here is the example of how to design functions
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

