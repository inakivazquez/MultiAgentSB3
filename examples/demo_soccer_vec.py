from ma_sb3.envs.soccer_v0 import SoccerEnv
from ma_sb3 import TimeLimitMAEnv

from gymnasium.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import pybullet as p


# Example of running the environment
if __name__ == "__main__":

    train = True
    #train = False
    load_previous_model = False

    algo = SAC

    if train:
        models = {}

        def make_env(env_id):
            def _init():
                ma_env = SoccerEnv(n_team_players=2, perimeter_side=10, render=False)
                ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=1000)
                agents_envs = ma_env.get_agents_envs()
                if env_id == 'red_0':
                    env= agents_envs['red_0']
                else:
                    env= agents_envs['blue_0']
                model = algo.load(f"policies/model_soccer_{algo.__name__}", env, tensorboard_log="./logs")
                models = {'soccer_player':model}
                ma_env.set_agent_models(models=models)
                return env
            return _init

        #envs = [make_env('red_0'), make_env('blue_0')]
        envs = [make_env('red_0')]

        # For parallel processing
        env = SubprocVecEnv(envs)

        if load_previous_model:
            model_player = algo.load(f"policies/model_soccer_{algo.__name__}", env, tensorboard_log="./logs")
        else:
            model_player = algo("MlpPolicy", env, verbose=1, tensorboard_log="./logs")

        total_timesteps_per_agent = 100_000
        training_iterations = 1
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        algo_name = model_player.__class__.__name__
        model_name = 'soccer_player_vec'

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            model_player.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{model_name}_{algo_name}")

        env.close()

        model_player.save(f"policies/model_soccer_{algo.__name__}")
        

    # TESTING SECTION
    ma_env = SoccerEnv(n_team_players=2, perimeter_side=10, render=True)
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
            #print("Actions:", actions)
            obs, rewards, terminated, truncated , _= ma_env.step_all(actions)

    ma_env.close()