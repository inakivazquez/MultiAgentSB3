from ma_sb3.envs.multipredator_prey_v1 import MultiPredatorPreyMAEnv
from ma_sb3 import TimeLimitMAEnv
from ma_sb3.utils import ma_train, ma_evaluate

from stable_baselines3 import PPO, SAC
import pybullet as p
import logging


# Example of running the environment
if __name__ == "__main__":

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    train = True
    #train = False
    load_previous_predator = True
    load_previous_prey = True

    n_predators = 3
    experiment_name = "dense"

    predator_algo = PPO
    prey_algo = PPO
    predator_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}
    prey_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}

    predator_model_path = f"policies/model_multipredator_{n_predators}preds_{predator_algo.__name__}_{experiment_name}"
    prey_model_path = f"policies/model_multiprey_{n_predators}preds_{prey_algo.__name__}_{experiment_name}"

    env_params = {'n_predators': n_predators, 'perimeter_side': 10, 'reward_all_predators': 10, 'reward_catching_predator': 0}

    if train:
        ma_env = MultiPredatorPreyMAEnv(**env_params, render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=250)

        model_algo_map = {'predator': (predator_algo, predator_algo_params), 'prey': (prey_algo, prey_algo_params)}

        models_to_load = {}
        if load_previous_predator:
            models_to_load['predator'] = predator_model_path
        if load_previous_prey:
            models_to_load['prey'] = prey_model_path

        trained_models = ma_train(ma_env, model_algo_map=model_algo_map,
                 models_to_train='__all__', models_to_load=models_to_load,
                 total_timesteps_per_model=1_000_000, training_iterations=100,
                 tb_log_suffix=f"{n_predators}preds_{experiment_name}")

        ma_env.close()

        trained_models['predator'].save(predator_model_path)
        trained_models['prey'].save(prey_model_path)
        
    # TESTING SECTION
    render = True
    record_video_file = None
    #record_video_file = "3preds_1prey.mp4"
    ma_env = MultiPredatorPreyMAEnv(**env_params, render=render, record_video_file=record_video_file)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=250)

    model_predator = predator_algo.load(predator_model_path)
    model_prey = prey_algo.load(prey_model_path)

    models = {'predator':model_predator, 'prey': model_prey}

    total_episodes = 100

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")
       