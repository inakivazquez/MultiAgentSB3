from ma_sb3.envs.soccer_v0 import SoccerEnv
from ma_sb3 import TimeLimitMAEnv

from stable_baselines3 import PPO, SAC
from gymnasium.wrappers.normalize import NormalizeObservation

import time

# Example of running the environment
if __name__ == "__main__":

    train = True
    #train = False
    load_previous_model = False

    algo = SAC

    n_team_players = 1

    if train:
        ma_env = SoccerEnv(n_team_players=n_team_players, perimeter_side=10, render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=1000)

        agents_envs = ma_env.get_agents_envs()
        models = {}

        env_red = agents_envs['red_0']
        env_blue = agents_envs['blue_0']

        env_red = NormalizeObservation(env_red)

        envs = {'red':env_red, 'blue':env_blue}
        envs = {'red':env_red}

        env_initial = env_red

        if load_previous_model:
            model_player = algo.load(f"policies/model_soccer_{algo.__name__}", env_initial, tensorboard_log="./logs")
        else:
            model_player = algo("MlpPolicy", env_initial, verbose=1, tensorboard_log="./logs")

        models = {'soccer_player':model_player}
        ma_env.set_agent_models(models=models)

        total_timesteps_per_agent = 50_000
        training_iterations = 1
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            for model_name, model_player in models.items():
                algo_name = model_player.__class__.__name__
                for env_name, env in envs.items():
                    print(f"Training the {env_name} team...")
                    model_player.set_env(env)
                    model_player.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{model_name}{n_team_players}p_{algo_name}")

        ma_env.close()

        model_player.save(f"policies/model_soccer_{algo.__name__}")
        

    # TESTING SECTION
    ma_env = SoccerEnv(n_team_players=n_team_players, perimeter_side=10, render=True)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=1000)

    model_player = algo.load(f"policies/model_soccer_{algo.__name__}")
 
    models = {'soccer_player':model_player}

    for _ in range(50):
        obs, info = ma_env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = models['soccer_player'].predict(agent_obs)[0]
            obs, rewards, terminated, truncated , _= ma_env.step_all(actions)
            print(actions['red_0'])


    ma_env.close()