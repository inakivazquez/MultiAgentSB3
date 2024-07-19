from ma_sb3.envs.multipredator_prey_ray_v0 import MultiPredatorPreyMAEnv
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
        ma_env = MultiPredatorPreyMAEnv(n_predators=1, perimeter_side=10, render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=100)

        agents_envs = ma_env.get_agents_envs()
        models = {}
        env_prey = agents_envs['prey']
        env_predator = None

        for agent_id, env in agents_envs.items(): 
            if agent_id.startswith('predator'):
                env_predator = env
                break

        if load_previous_predator:
            model_predator = predator_algo.load(f"policies/model_multipredator_{predator_algo.__name__}", env_predator, tensorboard_log="./logs")
        else:
            model_predator = predator_algo("MlpPolicy", env_predator, verbose=1, tensorboard_log="./logs")
        if load_previous_prey:
            model_prey = prey_algo.load(f"policies/model_multiprey_{prey_algo.__name__}", env_prey, tensorboard_log="./logs")
        else:
            model_prey = prey_algo("MlpPolicy", env_prey, verbose=1, tensorboard_log="./logs")

        models = {'predator':model_predator, 'prey': model_prey}
        ma_env.set_agent_models(models=models)

        total_timesteps_per_agent = 200_000
        training_iterations = 20
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            for model_name, model in models.items():
                algo_name = model.__class__.__name__
                model.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"ray_multi{model_name}_{algo_name}")

        ma_env.close()

        model_predator.save(f"policies/model_multipredator_{predator_algo.__name__}")
        model_prey.save(f"policies/model_multiprey_{prey_algo.__name__}")
        

    # TESTING SECTION
    ma_env = MultiPredatorPreyMAEnv(n_predators=1, perimeter_side=10, render=True)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=100)

    model_predator = predator_algo.load(f"policies/model_multipredator_{predator_algo.__name__}")
    model_prey = prey_algo.load(f"policies/model_multiprey_{prey_algo.__name__}")

    models = {'predator':model_predator, 'prey': model_prey}

    for _ in range(50):
        obs, info = ma_env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            actions = {}
            for agent_id, agent_obs in obs.items():
                model_id = 'predator' if agent_id.startswith('predator') else 'prey'
                actions[agent_id] = models[model_id].predict(agent_obs)[0]
            #print("Actions:", actions)
            obs, rewards, terminated, truncated , _= ma_env.step_all(actions)

    ma_env.close()