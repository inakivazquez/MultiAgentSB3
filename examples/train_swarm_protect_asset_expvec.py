from ma_sb3.envs.swarm_protect_asset_v5 import SwarmProtectAssetEnv
from ma_sb3 import TimeLimitMAEnv
from ma_sb3.utils import ma_train, ma_evaluate, ma_train_vec

from stable_baselines3 import PPO, SAC, TD3, DDPG, DQN
import logging
from stable_baselines3.common.env_util import make_vec_env
import argparse

import json

def save_parameters(hyperparams, env_params, file):
    """
    Save hyperparameters to a JSON file.
    """
    with open(file, 'w') as f:
        json.dump(hyperparams, f, indent=4)
        json.dump(env_params, f, indent=4)


# Example of running the environment
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate swarm shape environment.")
    parser.add_argument("-c", "--communication_items", type=int, default=0, help="Number of communication items.")
    parser.add_argument("-t", "--train", type=int, help="Train for the steps provided.")
    parser.add_argument("-l", "--load", action='store_true', help="Load previous policy and continue training.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    train = args.train is not None
    robot_load_previous_model = args.load
    if train:
        print("Training mode activated.")
    else:
        print("Testing mode activated.")
    if robot_load_previous_model:
        print("Loading previous model.")

    num_robots = 60
    num_learning_robots = 5
    num_assets = 2
    nrays = 20
    span_angle_degrees = 360
    communication_items = args.communication_items

    prefix = f"multi_EXPvec_3see_{num_learning_robots}l"

    experiment_name = f"{prefix}_c{communication_items}_{num_robots}a_{nrays}r_{span_angle_degrees}"
    seed = 42
    num_time_steps = args.train if train else 0

    robot_algo = SAC

    robot_algo_params = {'policy': "MlpPolicy",
                        'seed': seed,
                        'verbose': 1,
                        'batch_size': 512,
                        'policy_kwargs': {'net_arch': [256, 256, 128], 'use_sde':False},
                        'learning_rate': 3e-4,
                        'tensorboard_log': "./logs"}

    robot_model_path = f"policies/{prefix}_{robot_algo.__name__}_model_c{communication_items}"

    env_params = {'num_robots': num_robots,
                  'num_assets': num_assets,
                  'agent_speed': 0.1,
                  'forward_only': False,
                  'nrays': nrays,
                  'span_angle_degrees': span_angle_degrees,
                  'obs_body_prefixes': ['robot', 'asset', 'asset'], # We declare asset twice to cover the cases of asset surrouneded or not
                  'individual_comm_items': communication_items,
                  'env_comm_items': 1,
                  'comm_learning': False,
                  'surrounding_required': 0.90,
                  'asset_move_force': 0,
                  'verbose': False
                  }

    if train:
        save_parameters(robot_algo_params, env_params, robot_model_path + "_params.json")

        ma_env = SwarmProtectAssetEnv(**env_params, render_mode=None)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

        model_algo_map = {'robot': (robot_algo, robot_algo_params)}

        models_to_load = {}
        if robot_load_previous_model:
            models_to_load['robot'] = robot_model_path

        models_to_train = '__all__'

        trained_models = ma_train_vec(ma_env, model_algo_map=model_algo_map,
                 models_to_train=models_to_train, models_to_load=models_to_load,
                 num_learning_agents=num_learning_robots,
                 total_timesteps_per_model=num_time_steps, training_iterations=1,
                 tb_log_suffix=f"{experiment_name}")

        ma_env.close()

        trained_models['robot'].save(robot_model_path)
        if robot_algo in [SAC, TD3, DDPG, DQN]:
            trained_models['robot'].save_replay_buffer(robot_model_path + ".pkl")
        

    if train  == True:
        quit()

    # TESTING SECTION
    render_mode = 'human'
    ma_env = SwarmProtectAssetEnv(**env_params, render_mode=render_mode)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

    print(f"Loading policy {robot_model_path}...")
    model_robot = robot_algo.load(robot_model_path)
    models = {'robot':model_robot}

    total_episodes = 10

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes, verbose=True)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")

    ma_env.close()
       