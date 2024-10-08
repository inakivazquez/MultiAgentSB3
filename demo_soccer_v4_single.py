from ma_sb3.envs.soccer_v4 import SoccerEnv
from ma_sb3 import TimeLimitMAEnv
from ma_sb3.utils import ma_train, ma_evaluate

from gymnasium.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO, SAC
import pybullet as p
import logging


# Example of running the environment
if __name__ == "__main__":

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    train = True
    #train = False
    red_team_load_previous_model = True

    n_players_per_team = 0
    experiment_name = "single_random_ball_relgoal"

    red_team_algo = PPO
    red_team_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}

    red_team_model_path = f"policies/model_soccer_red_{n_players_per_team}p_{red_team_algo.__name__}_{experiment_name}"

    env_params = {'n_team_players': n_players_per_team, 'perimeter_side': 10}

    if train:
        ma_env = SoccerEnv(**env_params, render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

        model_algo_map = {'soccer_red': (red_team_algo, red_team_algo_params)}

        models_to_load = {}
        if red_team_load_previous_model:
            models_to_load['soccer_red'] = red_team_model_path

        trained_models = ma_train(ma_env, model_algo_map=model_algo_map,
                 models_to_train='all', models_to_load=models_to_load,
                 total_timesteps_per_model=200_000, training_iterations=1,
                 tb_log_suffix=f"{n_players_per_team}p_{experiment_name}")

        ma_env.close()

        trained_models['soccer_red'].save(red_team_model_path)
        
    # TESTING SECTION
    render = True
    record_video_file = None
    #record_video_file = "soccer.mp4"
    ma_env = SoccerEnv(**env_params, render=render, record_video_file=record_video_file)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

    model_red_team = red_team_algo.load(red_team_model_path)

    models = {'soccer_red':model_red_team}

    total_episodes = 100

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")
       