from ma_sb3.envs import PredatorPreyMAEnv
from ma_sb3 import TimeLimitMAEnv

from gymnasium.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO, SAC
import pybullet as p


# Example of running the environment
if __name__ == "__main__":

    train = True
    #train = False
    load_previous_predator = False
    load_previous_prey = False

    predator_algo = PPO
    prey_algo = PPO

    if train:
        ma_env = PredatorPreyMAEnv(render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=100)

        agents_envs = ma_env.get_agents_envs()
        
        #env_predator = TimeLimit(agents_envs['predator'], max_episode_steps=100)
        #env_prey = TimeLimit(agents_envs['prey'], max_episode_steps=100)

        env_predator = agents_envs['predator']
        env_prey = agents_envs['prey']

        if load_previous_predator:
            model_predator = predator_algo.load(f"policies/model_predator_{predator_algo.__name__}", env_predator, tensorboard_log="./logs")
        else:
            model_predator = predator_algo("MlpPolicy", env_predator, verbose=1, tensorboard_log="./logs")

        if load_previous_prey:
            model_prey = prey_algo.load(f"policies/model_prey_{prey_algo.__name__}", env_prey, tensorboard_log="./logs")
        else:
            model_prey = prey_algo("MlpPolicy", env_prey, verbose=1, tensorboard_log="./logs")

        ma_env.set_agent_models(models = {'predator':model_predator, 'prey': model_prey})

        total_timesteps_per_agent = 200_000
        training_iterations = 20
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            model_predator.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"predator_{predator_algo.__name__}")
            model_prey.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"prey_{prey_algo.__name__}")

        ma_env.close()

        model_predator.save(f"policies/model_predator_{predator_algo.__name__}")
        model_prey.save(f"policies/model_prey_{prey_algo.__name__}")
        

    # TESTING SECTION
    ma_env = PredatorPreyMAEnv(render=True)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=100)

    model_predator = predator_algo.load(f"policies/model_predator_{predator_algo.__name__}")
    model_prey = prey_algo.load(f"policies/model_prey_{prey_algo.__name__}")

    for _ in range(50):
        obs, info = ma_env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {
                "predator": model_predator.predict(obs['predator'])[0][0],
                "prey": model_prey.predict(obs['prey'])[0][0]
            }

            #print("Actions:", actions)
            obs, rewards, terminated, truncated , _= ma_env.step_all(actions)

    ma_env.close()