from ma_sb3.envs.soccer_single_v1 import SoccerSingleEnv
from ma_sb3 import TimeLimitMAEnv

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers.normalize import NormalizeObservation

import time

# Example of running the environment
if __name__ == "__main__":

    train = True
    #train = False
    load_previous_model = True

    algo = SAC

    n_team_players = 1

    if train:
        ma_env = SoccerSingleEnv(n_team_players=n_team_players, perimeter_side=10, render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=1000)

        agents_envs = ma_env.get_agents_envs()

        env_red = agents_envs['red_0']

        """hyperparams = {
            "batch_size": 256,
            "gamma": 0.9556,
            "learning_rate": 0.0005,
            "policy_kwargs": {
                "net_arch": [
                    32,
                    32
                ]
            },
            "use_sde": False
        }"""
        hyperparams = {}


        if load_previous_model:
            model = algo.load(f"policies/model_soccer_single_{algo.__name__}", env_red, tensorboard_log="./logs")
        else:
            model = algo("MlpPolicy", env_red, verbose=1, tensorboard_log="./logs", **hyperparams)

        models = {'soccer_single':model}
        ma_env.set_agent_models(models=models)

        total_timesteps_per_agent = 100_000
        training_iterations = 1
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            for model_name, model in models.items():
                algo_name = model.__class__.__name__
                print(f"Training the {model_name} model...")
                checkpoint_callback = CheckpointCallback(save_freq=steps_per_iteration, save_path=f"checkpoints/{model_name}{n_team_players}p_{algo_name}")
                model.learn(total_timesteps=steps_per_iteration, callback=checkpoint_callback, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{model_name}{n_team_players}p_{algo_name}")

        ma_env.close()

        model.save(f"policies/model_soccer_single_{algo.__name__}")
        

    # TESTING SECTION
    ma_env = SoccerSingleEnv(n_team_players=n_team_players, perimeter_side=10, render=True)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=1000)

    model = algo.load(f"policies/model_soccer_single_{algo.__name__}")

    models = {'soccer_single':model}

    for _ in range(50):
        obs, info = ma_env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = models['soccer_single'].predict(agent_obs, deterministic=True)[0]
            #print("Actions:", actions)
            obs, rewards, terminated, truncated , _= ma_env.step_all(actions)

            #print(", ".join([f"{num:.2f}" for num in obs['red_0']]))

    ma_env.close()
