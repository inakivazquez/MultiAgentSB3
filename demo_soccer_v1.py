from ma_sb3.envs.soccer_v1 import SoccerEnv
from ma_sb3 import TimeLimitMAEnv

from stable_baselines3 import PPO, SAC
from gymnasium.wrappers.normalize import NormalizeObservation

import time

# Example of running the environment
if __name__ == "__main__":

    train = True
    #train = False
    load_previous_model = False

    algo_red = PPO
    algo_blue = PPO

    n_team_players = 1

    if train:
        ma_env = SoccerEnv(n_team_players=n_team_players, perimeter_side=10, render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=1000)

        agents_envs = ma_env.get_agents_envs()

        env_red = agents_envs['red_0']
        env_blue = agents_envs['blue_0']

        env_red = NormalizeObservation(env_red)
        env_blue = NormalizeObservation(env_blue)

        if load_previous_model:
            model_red = algo_red.load(f"policies/model_soccer_{algo_red.__name__}", env_red, tensorboard_log="./logs")
            model_blue = algo_blue.load(f"policies/model_soccer_{algo_blue.__name__}", env_blue, tensorboard_log="./logs")
        else:
            model_red = algo_red("MlpPolicy", env_red, verbose=1, tensorboard_log="./logs")
            model_blue = algo_blue("MlpPolicy", env_blue, verbose=1, tensorboard_log="./logs")

        models = {'soccer_red':model_red, 'soccer_blue':model_blue}
        ma_env.set_agent_models(models=models)

        total_timesteps_per_agent = 100_000
        training_iterations = 1
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            for model_name, model in models.items():
                algo_name = model.__class__.__name__
                print(f"Training the {model_name} model...")
                if model_name == 'soccer_red':
                #if model_name == 'soccer_blue':
                    model.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{model_name}{n_team_players}p_{algo_name}")

        ma_env.close()

        model_red.save(f"policies/model_soccer_red_{algo_red.__name__}")
        model_red.save(f"policies/model_soccer_blue_{algo_blue.__name__}")
        

    # TESTING SECTION
    ma_env = SoccerEnv(n_team_players=n_team_players, perimeter_side=10, render=True)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=1000)

    model_red = algo_red.load(f"policies/model_soccer_{algo_red.__name__}")
    model_blue = algo_blue.load(f"policies/model_soccer_{algo_blue.__name__}")
 
    models = {'soccer_red':model_red, 'soccer_blue':model_blue}

    for _ in range(50):
        obs, info = ma_env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {}
            for agent_id, agent_obs in obs.items():
                model_id = 'soccer_red' if agent_id.startswith('red') else 'soccer_blue'
                actions[agent_id] = models[model_id].predict(agent_obs)[0]
            #print("Actions:", actions)
            obs, rewards, terminated, truncated , _= ma_env.step_all(actions)

    ma_env.close()