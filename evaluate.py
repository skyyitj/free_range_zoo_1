"""Evaluate wildfire baselines on testing configurations."""
# python evaluation.py eva_result /home/liuchi/zouyu/free-range-zoo/wildfire_config/WS1.pkl
import warnings
import logging.handlers
from datetime import datetime

warnings.simplefilter('ignore', UserWarning)

import argparse
import os
import sys
import torch
import pickle

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs.wildfire.baselines import NoopBaseline, RandomBaseline, StrongestBaseline, WeakestBaseline, GenerateAgent

FORMAT_STRING = "[%(asctime)s] [%(levelname)8s] [%(name)10s] [%(filename)21s:%(lineno)03d] %(message)s"

def setup_logging(output_dir: str) -> None:
    """设置日志系统，包括文件和控制台输出"""
    # 创建时间戳文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(output_dir, 'logs', timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # 创建主日志文件和测试日志文件的处理器
    main_file_handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
    test_file_handler = logging.FileHandler(os.path.join(log_dir, 'test.log'))
    console_handler = logging.StreamHandler()

    # 设置格式
    formatter = logging.Formatter(FORMAT_STRING)
    main_file_handler.setFormatter(formatter)
    test_file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 配置主日志记录器
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(main_file_handler)
    main_logger.addHandler(console_handler)

    # 配置测试日志记录器
    test_logger = logging.getLogger('baseline')
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(test_file_handler)
    test_logger.addHandler(console_handler)

def create_log_heading(heading: str) -> str:
    """Create a formatted log heading."""
    return f"\n{'=' * 10} {heading} {'=' * 10}\n"

# 添加create_log_heading方法到logging模块
logging.create_log_heading = create_log_heading

def main() -> None:
    """Run the training experiment."""
    global device, args, dataset
    args = handle_args()
    
    # 设置日志系统
    setup_logging(args.output)
    
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    if args.threads > 1:
        torch.set_num_threads(args.threads)

    os.makedirs(args.output, exist_ok=True)

    if os.path.exists(args.output):
        main_logger.warning(f'Output directory {args.output} already exists and may contain artifacts from a previous run.')

    main_logger.info(f'Running the baseline experiment on device {device} and parameters:')
    for key, value in vars(args).items():
        main_logger.info(f'- {key}: {value}')

    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator = generator.manual_seed(args.seed if args.seed is not None else 42)
    torch.manual_seed(torch.randint(0, 100000000, (1, ), generator=generator).item())

    main_logger.info('Initializing dataset')

    try:
        test()
    except KeyboardInterrupt:
        main_logger.warning('Testing interrupted by user')
    except Exception as e:
        main_logger.error(f'Error during testing: {e}')
        raise e


@torch.no_grad()
def test() -> None:
    """
    Run the testing episodes for the model.

    Args:
        model: nn.Module - The model to validate.
    """

    agents = ['firefighter_1', 'firefighter_2', 'firefighter_3']
    
    with open(args.config, "rb") as config_file:
        wildfire_configuration = pickle.load(config_file)
        
    env = wildfire_v0.parallel_env(
        parallel_envs=args.testing_episodes,
        max_steps=50,
        device=device,
        configuration=wildfire_configuration,
        buffer_size=50,
        single_seeding=True,
        show_bad_actions=False,
    )

    env = action_mapping_wrapper_v0(env)
    observation, _ = env.reset(seed=0)

    # 初始化每个智能体的策略
    agents = {}
    for agent_name in env.agents:
        # agents[agent_name] = NoopBaseline(env.action_space(agent_name), parallel_envs=args.testing_episodes)
        # agents[agent_name] = RandomBaseline(env.action_space(agent_name), parallel_envs=args.testing_episodes)
        # agents[agent_name] = StrongestBaseline(env.action_space(agent_name), parallel_envs=args.testing_episodes)
        # agents[agent_name] = WeakestBaseline(env.action_space(agent_name), parallel_envs=args.testing_episodes)
        #agents[agent_name] = HeuristicBaseline(env.action_space(agent_name), parallel_envs=args.testing_episodes)
        agents[agent_name] = GenerateAgent(env.action_space(agent_name), parallel_envs=args.testing_episodes,configuration = wildfire_configuration)
    step = 0
    total_rewards = {agent: torch.zeros(args.testing_episodes) for agent in agents}
    while not torch.all(env.finished):
        test_logger.info(f'STEP {step}')
        agent_actions = {}
        for agent_name, agent_model in agents.items():
            agent_model.observe(observation[agent_name])

            actions = agent_model.act(env.action_space(agent_name))
            actions = torch.tensor(actions, device=device, dtype=torch.int32)
            agent_actions[agent_name] = actions

        observation, reward, term, trunc, info = env.step(agent_actions)

        test_logger.info('ACTIONS')
        for batch in range(args.testing_episodes):
            batch_actions = ' '.join(f'{agent_name}: {str(agent_actions[agent_name].tolist()):<10}\t'
                                     for agent_name in env.agents)
            test_logger.info(f'{batch + 1}:\t{batch_actions}')

        test_logger.info('REWARDS')
        for agent_name in env.agents:
            test_logger.info(f'{agent_name}: {reward[agent_name]}')
            total_rewards[agent_name] += reward[agent_name]

        step += 1

    test_logger.info(logging.create_log_heading('TOTAL REWARDS'))
    for agent_name, total_reward in total_rewards.items():
        test_logger.info(f'{agent_name}: {total_reward}')

    totals = torch.zeros(args.testing_episodes)
    for agent, reward_tensor in total_rewards.items():
        totals += reward_tensor

    total_mean = round(totals.mean().item(), 3)
    total_std_dev = round(totals.std().item(), 3)

    average_mean = round(total_mean / len(agents), 3)
    average_std_dev = round(total_std_dev / len(agents), 3)

    test_logger.info(logging.create_log_heading('REWARD SUMMARY'))
    test_logger.info(f'Average Reward: {average_mean} ± {average_std_dev}')
    test_logger.info(f'Total Reward: {total_mean} ± {total_std_dev}')


def handle_args() -> argparse.Namespace:
    """
    Handle script arguments.

    Returns:
        argparse.Namespace - parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Run baseline policies on a given wildfire configuration.')

    general = parser.add_argument_group('General')
    general.add_argument('output', type=str, help='output directory for all experiments artifacts')
    general.add_argument('config', type=str, help='path to environment configuration to utilize')
    general.add_argument('--cuda', action='store_true', help='Utilize cuda if available')
    general.add_argument('--threads', type=int, default=1, help='utilize this many threads for the experiment')

    reproducible = parser.add_argument_group('Reproducibility')
    reproducible.add_argument('--seed', type=int, default=16, help='seed for the experiment')
    reproducible.add_argument('--dataset_seed', type=int, default=None, help='seed for initializing the configuration dataset')

    validation = parser.add_argument_group('Validation')
    validation.add_argument('--testing_episodes', type=int, default=16, help='number of episodes to run per test')

    return parser.parse_args()


# 初始化日志记录器
main_logger = logging.getLogger('main')
test_logger = logging.getLogger('baseline')

if __name__ == '__main__':
    main()
    sys.exit()