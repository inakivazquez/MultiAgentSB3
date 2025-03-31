from ma_sb3.envs.swarm_shape_v0 import SwarmShapeEnv
from ma_sb3 import TimeLimitMAEnv
from ma_sb3.utils import ma_train, ma_evaluate

from stable_baselines3 import PPO, SAC, TD3, DDPG, DQN
import pybullet as p
import logging
from stable_baselines3.common.env_util import make_vec_env
import argparse

# Example of running the environment
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate swarm shape environment.")
    parser.add_argument("-c", "--communication_items", type=int, default=0, help="Number of communication items.")
    parser.add_argument("-n", "--n_steps", type=int, default=100_000, help="Number of trainign steps.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    train = False
    cube_load_previous_model = True
    prexif = "V5"

    num_cubes = 16
    nrays = 20
    span_angle_degrees = 360
    communication_items = args.communication_items
    experiment_name = f"shape_{prexif}_c{communication_items}_{num_cubes}a_{nrays}r_{span_angle_degrees}"
    seed = 42
    num_time_steps = args.n_steps

    cube_algo = PPO

    cube_algo_params = {'policy': "MlpPolicy", 'seed': seed, 'verbose': 1, 'tensorboard_log': "./logs"}

    #cube_model_path = "policies/shape_current_model"
    cube_model_path = f"policies/shape_{prexif}_model_c{communication_items}"

    env_params = {'num_cubes': num_cubes,
                  'agent_speed': 0.1,
                  'forward_only': False,
                  'nrays': nrays,
                  'span_angle_degrees': span_angle_degrees,
                  'communication_items': communication_items}

    if train:
        ma_env = SwarmShapeEnv(**env_params, render_mode=None)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

        model_algo_map = {'cube': (cube_algo, cube_algo_params)}

        models_to_load = {}
        if cube_load_previous_model:
            models_to_load['cube'] = cube_model_path

        models_to_train = 'all'

        trained_models = ma_train(ma_env, model_algo_map=model_algo_map,
                 models_to_train=models_to_train, models_to_load=models_to_load,
                 total_timesteps_per_model=num_time_steps, training_iterations=1,
                 tb_log_suffix=f"{experiment_name}")

        ma_env.close()

        trained_models['cube'].save(cube_model_path)
        if cube_algo in [SAC, TD3, DDPG, DQN]:
            trained_models['cube'].save_replay_buffer(cube_model_path + ".pkl")
        

    if train  == True:
        quit()
    # TESTING SECTION
    render_mode = 'human'
    record_video_file = None
    ma_env = SwarmShapeEnv(**env_params, render_mode=render_mode)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

    model_cube = cube_algo.load(cube_model_path)
    models = {'cube':model_cube}

    total_episodes = 10

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes, verbose=True)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")
       