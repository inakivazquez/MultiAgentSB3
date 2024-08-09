from ma_sb3.envs.soccer_disc_v2 import SoccerEnv
from ma_sb3 import TimeLimitMAEnv

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers.normalize import NormalizeObservation

import time

# Example of running the environment
if __name__ == "__main__":

    train = True
    train = False
    load_previous_model = False

    algo = PPO

    n_team_players = 2

    if train:
        ma_env = SoccerEnv(n_team_players=n_team_players, perimeter_side=10, render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

        agents_envs = ma_env.get_agents_envs()

        env = agents_envs['0']

        env = NormalizeObservation(env)

        if load_previous_model:
            model = algo.load(f"policies/model_soccer_uni_{algo.__name__}", env, tensorboard_log="./logs")
        else:
            model = algo("MlpPolicy", env, verbose=1, tensorboard_log="./logs")

        models = {'soccer':model}
        ma_env.set_agent_models(models=models)

        total_timesteps_per_agent = 1_000_000
        training_iterations = 1
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            for model_name, model in models.items():
                algo_name = model.__class__.__name__
                print(f"Training the {model_name} model...")
                checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=f"checkpoints/{model_name}disc{n_team_players}p_{algo_name}")
                model.learn(total_timesteps=steps_per_iteration, callback=checkpoint_callback, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{model_name}disc{n_team_players}p_{algo_name}")

        ma_env.close()

        model.save(f"policies/model_soccer_uni_{algo.__name__}")
        

    # TESTING SECTION
    ma_env = SoccerEnv(n_team_players=n_team_players, perimeter_side=10, render=True)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=500)

    model = algo.load(f"policies/model_soccer_uni_{algo.__name__}")
    #model = algo.load(f"checkpoints/soccerdisc2p_PPO/rl_model_130000_steps.zip")

    models = {'soccer':model}

    for _ in range(50):
        obs, info = ma_env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {}
            for agent_id, agent_obs in obs.items():
                model_id = 'soccer'
                actions[agent_id] = models[model_id].predict(agent_obs, deterministic=True)[0]
            #print("Actions:", actions)
            obs, rewards, terminated, truncated , _= ma_env.step_all(actions)
            #print(obs)

    ma_env.close()
