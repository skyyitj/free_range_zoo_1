@dataclass
class WildfireState:
    """野火环境状态表示"""
    fires: torch.Tensor  # 野火存在矩阵 <batch, y, x>，0表示无火，正数表示有火，负数表示已熄灭
    intensity: torch.Tensor  # 野火强度矩阵 <batch, y, x>，范围[0,num_fire_states-1]
    fuel: torch.Tensor  # 燃料剩余矩阵 <batch, y, x>，范围[0,initial_fuel]
    agents: torch.Tensor  # 智能体位置矩阵 <batch, agent, 2>，每个元素为(y, x)坐标
    suppressants: torch.Tensor  # 智能体灭火剂矩阵 <batch, agent>，范围[0,capacity]
    capacity: torch.Tensor  # 智能体最大灭火剂矩阵 <batch, agent>，范围通常为[1,3]
    equipment: torch.Tensor  # 智能体设备状态矩阵 <batch, agent>，范围[0,1]，1表示完好
    attack_counts: torch.Tensor  # 每个格子的灭火剂使用次数 <batch, y, x>，范围[0,∞)


@dataclass
class FireConfiguration:
    """野火配置"""
    num_fire_states: int  # 火焰强度状态数量，通常为5
    intensity_increase_probability: float  # 火焰强度增加概率，范围[0,1]，通常为1.0
    intensity_decrease_probability: float  # 火焰强度降低概率，范围[0,1]，通常为0.8
    extra_power_decrease_bonus: float  # 额外灭火能力加成，范围[0,1]，通常为0.12
    burnout_probability: float  # 火焰自然熄灭概率，范围[0,1]
    base_spread_rate: float  # 基础蔓延速率，范围[0,∞)，值越大蔓延越快
    max_spread_rate: float  # 最大蔓延速率，范围[base_spread_rate,∞)，通常为67.0
    random_ignition_probability: float  # 随机点火概率，范围[0,1]，通常为0.0
    cell_size: float  # 单元格大小，范围(0,∞)，通常为200.0
    wind_direction: float  # 风向，范围[0,2π]，以弧度表示
    initial_fuel: int  # 初始燃料量，范围[0,∞)，通常为2


@dataclass
class AgentConfiguration:
    """智能体配置"""
    fire_reduction_power: torch.Tensor  # 灭火能力，范围(0,3)，决定每次攻击减少的火焰强度，通常为1
    attack_range: torch.Tensor  # 攻击范围，范围[0,3)，决定智能体可攻击的距离，通常为1
    suppressant_states: int  # 灭火剂状态数量，范围[1,10)，通常为3
    initial_suppressant: int  # 初始灭火剂数量，范围[0,suppressant_states-1]，通常为2
    suppressant_decrease_probability: float  # 灭火剂减少概率，范围[0,1]，通常为1/3
    suppressant_refill_probability: float  # 灭火剂补充概率，范围[0,1]，通常为1/3
    initial_equipment_state: float  # 初始设备状态，范围[0,2]，通常为2（最佳状态）
    repair_probability: float  # 设备修复概率，范围[0,1]，通常为1.0
    degrade_probability: float  # 设备退化概率，范围[0,1]，通常为1.0
    critical_error_probability: float  # 关键错误概率，范围[0,1]，通常为0.0
    tank_switch_probability: float  # 水箱切换概率，范围[0,1]，通常为1.0


@dataclass
class RewardConfiguration:
    """奖励配置"""
    fire_rewards: torch.Tensor  # 扑灭野火奖励，根据火焰类型不同而不同，通常为[0,20.0,50.0,20.0]
    bad_attack_penalty: float  # 错误攻击惩罚（攻击无火区域），通常为-100.0
    burnout_penalty: float  # 火焰自然熄灭惩罚（未成功扑灭），通常为-1.0
    termination_reward: float  # 环境终止奖励，通常为0.0


# 动作空间定义
def build_action_space(num_tasks):
    """
    构建可用动作空间

    每个智能体的动作由两部分组成：
    - 第一部分：选择攻击目标（火焰位置）
    - 第二部分：选择攻击方式（0表示使用灭火剂，-1表示无操作）
    """
    # 可选的动作有：攻击特定火焰（索引0至num_tasks-1）或者无操作(-1)
    return gymnasium.spaces.MultiDiscrete([num_tasks, 2])


# 关键转换类
class FireDecreaseTransition(nn.Module):
    """
    处理火焰强度减少的转换函数

    当智能体使用灭火剂时，会调用此函数降低火焰强度
    参数:
        fire_shape: 火焰张量形状
        stochastic_decrease: 是否使用随机减少逻辑
        decrease_probability: 火焰强度减少基础概率
        extra_power_decrease_bonus: 额外灭火能力加成
    """

    def forward(self, state, attack_counts, randomness_source, return_put_out=False):
        """
        更新火焰强度

        当攻击计数满足要求时，根据概率降低火焰强度
        如果fire_decrease_mask生效，则对应位置的火焰强度-1
        当火焰强度降至0，火焰被熄灭，state.fires对应位置乘以-1
        """
        # 获取需要的灭火剂数量
        required_suppressants = torch.where(state.fires >= 0, state.fires, torch.zeros_like(state.fires))
        # 计算攻击差值（负值表示已满足灭火需求）
        attack_difference = required_suppressants - attack_counts

        # 找出已经满足灭火需求的点燃格子
        lit_tiles = torch.logical_and(state.fires > 0, state.intensity > 0)
        suppressant_needs_met = torch.logical_and(attack_difference <= 0, lit_tiles)

        # 根据配置计算降低概率
        if self.stochastic_decrease:
            self.decrease_probabilities[suppressant_needs_met] = self.decrease_probability + \
                                                                 -1 * attack_difference[
                                                                     suppressant_needs_met] * self.extra_power_decrease_bonus
        else:
            self.decrease_probabilities[suppressant_needs_met] = 1.0

        # 应用随机性决定哪些火焰强度降低
        fire_decrease_mask = randomness_source < self.decrease_probabilities
        state.intensity[fire_decrease_mask] -= 1

        # 检查是否熄灭
        just_put_out = torch.logical_and(fire_decrease_mask, state.intensity <= 0)
        state.fires[just_put_out] *= -1
        state.fuel[just_put_out] -= 1

        return (state, just_put_out) if return_put_out else state


# 火焰增加逻辑
class FireIncreaseTransition(nn.Module):
    """
    处理火焰强度增加的转换函数

    当智能体没有满足灭火要求时，火焰强度会增加
    参数:
        fire_shape: 火焰张量形状
        fire_states: 火焰状态数量
        stochastic_increase: 是否使用随机增加逻辑
        intensity_increase_probability: 火焰强度增加基础概率
        stochastic_burnouts: 是否使用随机燃尽逻辑
        burnout_probability: 燃尽概率
    """

    def forward(self, state, attack_counts, randomness_source, return_burned_out=False):
        """
        更新火焰强度

        如果没有满足灭火要求，火焰强度会增加
        如果火焰强度达到最大值，火焰会烧尽
        """
        self._reset_buffers()

        # 获取需要的灭火剂数量
        required_suppressants = torch.where(state.fires >= 0, state.fires, torch.zeros_like(state.fires))
        attack_difference = required_suppressants - attack_counts

        # 找出未满足灭火需求的点燃格子
        lit_tiles = torch.logical_and(state.fires > 0, state.intensity > 0)
        suppressant_needs_unmet = torch.logical_and(attack_difference > 0, lit_tiles)

        # 检查即将燃尽的格子和正在燃烧的格子
        almost_burnouts = torch.logical_and(suppressant_needs_unmet, state.intensity == self.almost_burnout_state)
        increasing = torch.logical_and(suppressant_needs_unmet, ~almost_burnouts)

        # 应用不同的概率逻辑
        if self.stochastic_increase:
            self.increase_probabilities[increasing] = self.intensity_increase_probability
        else:
            self.increase_probabilities[increasing] = 1.0

        if self.stochastic_burnouts:
            self.increase_probabilities[almost_burnouts] = self.burnout_probability
        else:
            self.increase_probabilities[almost_burnouts] = self.intensity_increase_probability

        # 应用随机性
        self.increase_probabilities = torch.clamp(self.increase_probabilities, 0, 1)
        fire_increase_mask = randomness_source < self.increase_probabilities

        # 更新火焰强度
        state.intensity[fire_increase_mask] += 1

        # 检查是否有火焰燃尽
        just_burned_out = torch.logical_and(fire_increase_mask, state.intensity >= self.burnout_state)
        state.fires[just_burned_out] *= -1
        state.fuel[just_burned_out] = 0

        return (state, just_burned_out) if return_burned_out else state


# 火焰蔓延逻辑
class FireSpreadTransition(nn.Module):
    """
    处理火焰蔓延的转换函数

    基于Eck等人2020年的火焰蔓延模型
    参数:
        fire_spread_weights: 火焰蔓延滤波器权重
        ignition_temperatures: 点火温度
        use_fire_fuel: 是否使用燃料系统
    """

    def forward(self, state, randomness_source):
        """
        更新火焰蔓延状态

        使用卷积操作计算每个格子的蔓延概率，并根据随机源决定是否蔓延
        """
        # 找出正在燃烧的格子
        lit = torch.logical_and(state.fires > 0, state.intensity > 0)
        lit = lit.to(torch.float32).unsqueeze(1)

        # 计算火焰蔓延概率（使用卷积滤波器）
        fire_spread_probabilities = self.fire_spread_filter(lit).squeeze(1)

        # 找出可以被点燃的格子（未点燃且有燃料）
        unlit_tiles = torch.logical_and(state.fires < 0, state.intensity == 0)
        if self.use_fire_fuel:
            unlit_tiles = torch.logical_and(unlit_tiles, state.fuel > 0)

        # 将不可点燃格子的概率设为0
        fire_spread_probabilities[~unlit_tiles] = 0

        # 应用随机性决定哪些格子被点燃
        fire_spread_mask = randomness_source < fire_spread_probabilities

        # 更新被点燃格子的状态
        state.fires[fire_spread_mask] *= -1  # 变为正值表示有火
        state.intensity[fire_spread_mask] = self.ignition_temperatures.expand(state.fires.shape[0], -1, -1)[
            fire_spread_mask]

        return state


# 设备状态逻辑
class EquipmentTransition(nn.Module):
    """
    处理设备状态的转换函数

    设备状态会影响灭火剂补充和灭火效率
    参数:
        agent_shape: 智能体张量形状
        stochastic_repair: 是否使用随机修复逻辑
        repair_probability: 修复概率
        stochastic_degrade: 是否使用随机退化逻辑
        degrade_probability: 退化概率
        critical_error_probability: 严重错误概率
        tank_switch_probability: 水箱切换概率
    """

    def forward(self, state):
        """
        更新设备状态

        根据配置更新设备的修复和退化状态
        """
        # 重置缓冲区
        self._reset_buffers()

        # 根据配置应用随机修复逻辑
        if self.stochastic_repair:
            self.repair_mask = torch.logical_and(
                state.equipment < 2,
                self.repair_randomness < self.repair_probability
            )
        else:
            self.repair_mask = state.equipment < 2

        # 根据配置应用随机退化逻辑
        if self.stochastic_degrade:
            self.degrade_mask = torch.logical_and(
                state.equipment > 0,
                self.degrade_randomness < self.degrade_probability
            )

            self.critical_mask = torch.logical_and(
                self.degrade_mask,
                self.critical_randomness < self.critical_error_probability
            )

            self.tank_mask = torch.logical_and(
                self.degrade_mask,
                ~self.critical_mask,
                self.tank_randomness < self.tank_switch_probability
            )
        else:
            self.degrade_mask = state.equipment > 0
            self.critical_mask = torch.full_like(self.degrade_mask, False, dtype=torch.bool)
            self.tank_mask = self.degrade_mask

        # 应用修复和退化逻辑
        state.equipment[self.repair_mask] += 1
        state.equipment[self.critical_mask] = 0
        state.equipment[self.tank_mask] -= 1

        return state


# 容量逻辑
class CapacityTransition(nn.Module):
    """
    处理灭火剂容量的转换函数

    根据设备状态更新灭火剂容量
    参数:
        capacity_bonuses: 容量加成配置
    """

    def forward(self, state):
        """
        更新灭火剂容量

        根据设备状态确定智能体的最大灭火剂容量
        """
        equipment_states = state.equipment.flatten().unsqueeze(1)
        capacity_bonus = self.capacity_bonuses[equipment_states].reshape(state.equipment.shape)

        state.capacity = self.base_capacity.expand(state.equipment.shape[0], -1) + capacity_bonus

        # 确保灭火剂不超过容量
        state.suppressants = torch.clamp(state.suppressants, max=state.capacity)

        return state


# 环境步进逻辑
def step_environment(self):
    """
    执行环境的一步更新

    处理顺序:
    1. 处理每个智能体的动作（攻击火焰）
    2. 更新野火状态（蔓延、强度变化）
    3. 更新智能体状态（灭火剂补充、设备状态）
    4. 计算奖励并返回
    """
    # 初始化各种变量和随机源
    rewards = {agent: torch.zeros(self.parallel_envs, device=self.device) for agent in self.agents}
    put_out_fires = torch.zeros((self.parallel_envs, self.max_y, self.max_x), device=self.device, dtype=torch.bool)

    # 处理智能体动作
    for agent in self.agents:
        agent_idx = self.agent_name_mapping[agent]
        action_task = self.actions[agent][:, 0]
        action_type = self.actions[agent][:, 1]

        # 检查是否合法攻击（是否有灭火剂、是否在范围内等）
        valid_attack = self._check_valid_attacks(agent_idx, action_task, action_type)

        # 更新攻击计数
        self._update_attack_counts(agent_idx, action_task, action_type, valid_attack)

        # 计算奖励
        rewards[agent] = self._calculate_rewards(agent_idx, action_task, action_type, valid_attack)

    # 生成随机源
    randomness = self.generator.generate(
        keys=["fire_decrease", "fire_increase", "fire_spread", "suppressant_decrease", "suppressant_refill"]
    )

    # 应用火焰减少转换
    self._state, fires_put_out = self.fire_decrease_transition(
        self._state,
        self._state.attack_counts,
        randomness["fire_decrease"],
        return_put_out=True
    )
    put_out_fires = torch.logical_or(put_out_fires, fires_put_out)

    # 应用火焰增加转换
    self._state, fires_burned_out = self.fire_increase_transition(
        self._state,
        self._state.attack_counts,
        randomness["fire_increase"],
        return_burned_out=True
    )

    # 应用火焰蔓延转换
    self._state = self.fire_spread_transition(
        self._state,
        randomness["fire_spread"]
    )

    # 更新灭火剂状态
    self._state = self.suppressant_decrease_transition(
        self._state,
        randomness["suppressant_decrease"]
    )

    # 补充灭火剂
    self._state, refill_mask = self.suppressant_refill_transition(
        self._state,
        randomness["suppressant_refill"],
        return_increased=True
    )

    # 更新设备状态
    self._state = self.equipment_transition(self._state)

    # 更新容量
    self._state = self.capacity_transition(self._state)

    # 调整奖励（惩罚自然熄灭、奖励成功扑灭等）
    self._adjust_rewards(rewards, fires_burned_out, put_out_fires)

    return rewards, self.terminations, self.infos


# 环境重置方法
def reset(self, seed=None, options=None):
    """
    重置环境到初始状态

    参数:
        seed: 随机种子
        options: 重置选项
    返回:
        observations: 初始观察
        infos: 附加信息
    """
    # 设置随机种子
    if seed is not None:
        self.seed(seed)

    # 初始化智能体动作到任务的映射
    self.agent_action_to_task_mapping = {agent: {} for agent in self.agents}
    self.fire_reduction_power = self.agent_config.fire_reduction_power
    self.suppressant_states = self.agent_config.suppressant_states

    # 初始化环境状态
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
        agents=torch.clone(self.agent_config.agents),
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
        attack_counts=torch.zeros(
            (self.parallel_envs, self.max_y, self.max_x),
            dtype=torch.float32,
            device=self.device,
        ),
    )

    # 如果提供了初始状态，则使用提供的状态
    if options is not None and options.get('initial_state') is not None:
        initial_state = options['initial_state']
        if len(initial_state) != self.parallel_envs:
            raise ValueError("初始状态的批次数必须与并行环境数相同")
        self._state = initial_state
    else:
        # 否则，根据配置初始化火焰和智能体状态
        lit_mask = self.fire_config.lit_mask if hasattr(self.fire_config, 'lit_mask') else None
        fire_types = self.fire_config.fire_types if hasattr(self.fire_config, 'fire_types') else torch.ones_like(
            self._state.fires[0])

        if lit_mask is not None:
            # 设置点燃的和未点燃的格子
            self._state.fires[:, lit_mask] = fire_types[lit_mask]
            self._state.fires[:, ~lit_mask] = -1 * fire_types[~lit_mask]
            # 设置点燃格子的火焰强度
            self._state.intensity[:, lit_mask] = self.fire_config.ignition_temp if hasattr(self.fire_config,
                                                                                           'ignition_temp') else torch.ones_like(
                self._state.intensity[:, lit_mask])
        else:
            # 如果没有提供lit_mask，则使用随机初始化
            # 此处可以根据需要实现随机初始化逻辑
            pass

        # 设置燃料
        self._state.fuel[self._state.fires != 0] = self.fire_config.initial_fuel

        # 设置智能体的初始状态
        self._state.suppressants[:, :] = self.agent_config.initial_suppressant
        self._state.capacity[:, :] = self.agent_config.initial_capacity if hasattr(self.agent_config,
                                                                                   'initial_capacity') else torch.ones_like(
            self._state.capacity)
        self._state.equipment[:, :] = self.agent_config.initial_equipment_state

    # 保存初始状态以便后续可能的重置
    if hasattr(self._state, 'save_initial'):
        self._state.save_initial()

    # 初始化奖励
    self.fire_rewards = self.reward_config.fire_rewards.unsqueeze(0).expand(self.parallel_envs,
                                                                            -1) if self.reward_config.fire_rewards.dim() == 1 else self.reward_config.fire_rewards
    self.num_burnouts = torch.zeros(self.parallel_envs, dtype=torch.int32, device=self.device)

    # 初始化任务（火焰位置）
    self.tasks = self._get_tasks()

    # 初始化智能体观察和动作空间
    observations = self._get_observations()
    self.action_spaces = {agent: build_action_space(len(self.tasks[i])) for i, agent in enumerate(self.agents)}

    # 初始化终止状态和信息
    self.terminations = {agent: torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device) for agent in
                         self.agents}
    self.infos = {agent: {"legal_moves": self.action_spaces[agent]} for agent in self.agents}

    return observations, self.infos



@dataclass
class WildfireObservation:
    """野火环境观察表示

    属性:
        fires: 野火存在矩阵 <z, y, x>，0表示无火，1表示有火
        intensity: 野火强度矩阵 <z, y, x>，范围[0,1]
        fuel: 燃料剩余矩阵 <z, y, x>，范围[0,100]
        agents: 智能体位置矩阵 <agent, 2>，每个元素为(y, x)坐标
        suppressants: 智能体灭火剂矩阵 <agent, 1>，范围[0,5]
        capacity: 智能体最大灭火剂矩阵 <agent, 1>，范围[0,5]
        equipment: 智能体设备状态矩阵 <agent, 1>，范围[0,1]
        attack_counts: 每个格子的灭火剂使用次数 <z, y, x>，范围[0,30]
    """
    fires: np.ndarray
    intensity: np.ndarray
    fuel: np.ndarray
    agents: np.ndarray
    suppressants: np.ndarray
    capacity: np.ndarray
    equipment: np.ndarray
    attack_counts: np.ndarray


class Env:
    def compute_observations(self, agent_id):
        agent = self.agent_list[agent_id]

        # 获取自身观察
        self_obs = self._get_self_observation(agent)

        # 获取其他智能体观察
        others_obs = self._get_others_observation(agent_id)

        # 获取野火任务观察
        tasks_obs = self._get_tasks_observation()

        # 组合所有观察
        observation = {
            'self': self_obs,
            'others': others_obs,
            'tasks': tasks_obs
        }

        return observation

    def _get_self_observation(self, agent):
        """获取自身观察"""
        # 位置坐标
        pos = agent.final_pos

        # 根据配置决定是否包含能量和灭火剂
        obs = [pos[0], pos[1]]  # 基础位置信息

        if self.include_power:
            obs.append(agent.power / self.max_power)  # 能量水平

        if self.include_suppressant:
            obs.append(agent.suppressant / self.max_suppressant)  # 灭火剂水平

        return np.array(obs, dtype=np.float32)

    def _get_others_observation(self, agent_id):
        """获取其他智能体的观察"""
        others_obs = []
        for other_id in range(self.agent_n):
            if other_id == agent_id:
                continue
            other_agent = self.agent_list[other_id]
            other_obs = self._get_self_observation(other_agent)
            others_obs.append(other_obs)
        return tuple(others_obs)

    def _get_tasks_observation(self):
        """获取野火任务观察"""
        tasks_obs = []
        for fire_id in range(self.num_tasks):
            fire = self.fire_list[fire_id]
            # 野火观察包含：位置(y,x)、等级、强度
            fire_obs = [
                fire.position[0],  # y坐标
                fire.position[1],  # x坐标
                fire.level,  # 野火等级
                fire.intensity  # 野火强度
            ]
            tasks_obs.append(np.array(fire_obs, dtype=np.float32))
        return tuple(tasks_obs)

    class Agent(ABC):
        """Generic interface for agents."""

        def __init__(self, agent_name: str, parallel_envs: int) -> None:
            """
            Initialize the agent.

            Args:
                agent_name: str - Name of the subject agent
                parallel_envs: int - Number of parallel environments to operate on
            """
            self.agent_name = agent_name
            self.parallel_envs = parallel_envs

        def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
            """
            Return a list of actions, one for each parallel environment.

            Args:
                action_space: free_range_rust.Space - Current action space available to the agent.
            Returns:
                List[List[int, int]] - List of actions, one for each parallel environment.
            """
            pass

        def observe(self, observation: Dict[str, Any]) -> None:
            """
            Observe the environment.

            Args:
                observation: Dict[str, Any] - Current observation from the environment.
            """
            pass

