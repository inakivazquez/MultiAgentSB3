from ma_sb3.envs.multipredator_prey_v1 import MultiPredatorPreyMAEnv
from ma_sb3 import TimeLimitMAEnv
from ma_sb3.utils import ma_train, ma_evaluate

from gymnasium.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO, SAC


# Example of running the environment
if __name__ == "__main__":

    n_predators = 3

    # TRAINING SECTION
    predator_algo = PPO
    predator_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}

    prey_algo = PPO
    prey_algo_params = {'policy': "MlpPolicy", 'verbose': 1, 'tensorboard_log': "./logs"}

    env_params = {'n_predators': n_predators, 'perimeter_side': 10, 'reward_all_predators': 10, 'reward_catching_predator': 0}

    ma_env = MultiPredatorPreyMAEnv(**env_params, render=False)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=250)

    model_algo_map = {'predator': (predator_algo, predator_algo_params), 'prey': (prey_algo, prey_algo_params)}

    trained_models = ma_train(ma_env, model_algo_map=model_algo_map, models_to_train='__all__',
                total_timesteps_per_model=10_000, training_iterations=5,
                tb_log_suffix=f"predator_prey")

    ma_env.close()

    trained_models['predator'].save("policies/predator_model")
    trained_models['prey'].save("policies/prey_model")
        
    # TESTING SECTION
    render = True
    record_video_file = None
    ma_env = MultiPredatorPreyMAEnv(**env_params, render=render, record_video_file=record_video_file)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=250)

    model_predator = predator_algo.load("policies/predator_model")
    model_prey = prey_algo.load("policies/prey_model")

    models = {'predator':model_predator, 'prey': model_prey}

    total_episodes = 100

    print(f"Evaluating models for {total_episodes} episodes...")
    avg_agent, avg_model = ma_evaluate(ma_env, models, total_episodes=total_episodes)

    print(f"Average rewards per agent:\n {avg_agent}")
    print(f"Average rewards per model:\n {avg_model}")
       