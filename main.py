from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
import torch
import pickle
from free_range_zoo.envs.wildfire.baselines import RandomBaseline

config_path = "./archive/competition_configs/wildfire/WS1.pkl"

log_dir = "./outputs/"
device = torch.device('cpu')

# 打开文件并使用 pickle.load 读取配置
with open(config_path, "rb") as file:
    wildfire_configuration = pickle.load(file)
    print(wildfire_configuration)

env = wildfire_v0.parallel_env(
    max_steps=100,
    parallel_envs=1,
    configuration=wildfire_configuration,
    device=device,
    log_directory=log_dir
)
env.reset()
env = action_mapping_wrapper_v0(env)
observations, infos = env.reset()
print("******************************observations:",observations)

agents = {agent_name: RandomBaseline(env.action_space(agent_name), parallel_envs=1) for agent_name in env.agents}

step = 0
episode_reward = {agent: 0.0 for agent in env.agents}

while not torch.all(env.finished):
    step += 1
    for agent_name, agent in agents.items():
        agent.observe(observations[agent_name][0])

    agent_actions = {
        agent_name: agents[agent_name].act(action_space=env.action_space(agent_name))
        for agent_name in env.agents
    }
    print("*******************************agent_actions:", agent_actions)

    next_observations, rewards, terminations, truncations, infos = env.step(agent_actions)
    ######learning
    # for agent_name, agent in agents.items():
    #     agent.learn(
    #         observation=observations[agent_name][0],
    #         action=agent_actions[agent_name],
    #         reward=rewards[agent_name],
    #         next_observation=next_observations[agent_name][0],
    #         done=terminations[agent_name]
    #     )
    for agent_name in env.agents:
        episode_reward[agent_name] += rewards[agent_name].sum().item()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!the reward of",agent_name,"is:",rewards[agent_name])

    observations = next_observations

print(f"Episode finished in {step} steps. Total rewards: {episode_reward}")


env.close()

print("ok!")
