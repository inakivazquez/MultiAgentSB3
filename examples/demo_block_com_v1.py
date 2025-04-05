from ma_sb3.envs.multiblock_push_ray_com_v1 import MultiBlockPushRayCom
from ma_sb3 import TimeLimitMAEnv
from ma_sb3.utils import ma_train, ma_evaluate

from stable_baselines3 import PPO, SAC, TD3, DDPG, DQN
import pybullet as p
import logging
from stable_baselines3.common.env_util import make_vec_env

# Example of running the environment
if __name__ == "__main__":

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    train = True
    cube_load_previous_model = True

    num_cubes = 16
    num_blocks = 4
    block_density = 5000
    nrays = 20
    span_angle_degrees = 360
    communication_items = 1
    experiment_name = f"comm{communication_items}_{num_cubes}a_{num_blocks}b_{nrays}r_{span_angle_degrees}_den{block_density}"
    seed = 42

    cube_algo = PPO

    cube_algo_params = {'policy': "MlpPolicy", 'seed': seed, 'verbose': 1, 'tensorboard_log': "./logs"}

    #cube_algo_params = {'policy': "MlpPolicy", 'seed': seed, 'verbose': 1, 'tensorboard_log': "./logs", 
    #                    'policy_kwargs':{'net_arch':[256, 256]}}

    #cube_model_path = f"policies/model_cubes_{cube_algo.__name__}_{experiment_name}"
    cube_model_path = "policies/current_model"

    env_params = {'num_cubes': num_cubes,
                  'num_blocks': num_blocks,
                  'nrays': nrays,
                  'span_angle_degrees': span_angle_degrees,
                  'block_density': block_density,
                  'communication_items': communication_items}

    if train:
        ma_env = MultiBlockPushRayCom(**env_params, render_mode=None)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

        model_algo_map = {'cube': (cube_algo, cube_algo_params)}

        models_to_load = {}
        if cube_load_previous_model:
            models_to_load['cube'] = cube_model_path

        models_to_train = '__all__'

        trained_models = ma_train(ma_env, model_algo_map=model_algo_map,
                 models_to_train=models_to_train, models_to_load=models_to_load,
                 total_timesteps_per_model=1_000_000, training_iterations=10,
                 tb_log_suffix=f"{experiment_name}")

        ma_env.close()

        trained_models['cube'].save(cube_model_path)
        if cube_algo in [SAC, TD3, DDPG, DQN]:
            trained_models['cube'].save_replay_buffer(cube_model_path + ".pkl")
        
    # TESTING SECTION
    render_mode = 'human'
    record_video_file = None
    ma_env = MultiBlockPushRayCom(**env_params, render_mode=render_mode)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

    model_cube = cube_algo.load(cube_model_path)
    models = {'cube':model_cube}

    total_episodes = 100

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes, verbose=True)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")
       