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
    load_previous_prey = True

    predator_algo = PPO
    prey_algo = PPO

    n_predators = 1

    predator_model_path = f"policies/model_multi_ray_predator_{n_predators}preds_{predator_algo.__name__}"
    prey_model_path = f"policies/model_multi_ray_prey_{n_predators}preds_{prey_algo.__name__}"

    if train:
        ma_env = MultiPredatorPreyMAEnv(n_predators=n_predators, perimeter_side=10, render=False)
        ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=250)

        agents_envs = ma_env.get_agents_envs()
        models = {}
        env_prey = agents_envs['prey']
        env_predator = None

        for agent_id, env in agents_envs.items(): 
            if agent_id.startswith('predator'):
                env_predator = env
                break

        if load_previous_predator:
            model_predator = predator_algo.load(predator_model_path, env_predator, tensorboard_log="./logs")
        else:
            model_predator = predator_algo("MlpPolicy", env_predator, verbose=1, tensorboard_log="./logs")
        if load_previous_prey:
            model_prey = prey_algo.load(prey_model_path, env_prey, tensorboard_log="./logs")
        else:
            model_prey = prey_algo("MlpPolicy", env_prey, verbose=1, tensorboard_log="./logs")

        models = {'predator':model_predator, 'prey': model_prey}
        ma_env.set_agent_models(models=models)

        total_timesteps_per_agent = 300_000
        training_iterations = 1
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        models_to_train = ['predator']

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            for model_name, model in models.items():
                if model_name in models_to_train:
                    algo_name = model.__class__.__name__
                    model.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"ray_{model_name}_{n_predators}preds_{algo_name}")

        ma_env.close()

        model_predator.save(predator_model_path)
        model_prey.save(prey_model_path)
        

    # TESTING SECTION
    ma_env = MultiPredatorPreyMAEnv(n_predators=n_predators, perimeter_side=10, render=True)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=250)

    model_predator = predator_algo.load(predator_model_path)
    model_prey = prey_algo.load(prey_model_path)

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