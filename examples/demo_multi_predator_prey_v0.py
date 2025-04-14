from ma_sb3.envs import MultiPredatorPreyMAEnv
from ma_sb3 import TimeLimitMAEnv

from stable_baselines3 import PPO, SAC
import pybullet as p


# Example of running the environment
if __name__ == "__main__":

    train = True
    #train = False
    load_previous_predator = True
    load_previous_prey = True

    predator_algo = PPO
    prey_algo = PPO

    n_predators = 4

    if train:
        ma_env = MultiPredatorPreyMAEnv(n_predators=n_predators, max_speed_predator=1, max_speed_prey=1, perimeter_side=10, render=False)
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
            model_predator = predator_algo.load(f"policies/model_multipredator_{predator_algo.__name__}", env_predator, tensorboard_log="./logs")
        else:
            model_predator = predator_algo("MlpPolicy", env_predator, verbose=1, tensorboard_log="./logs")
        if load_previous_prey:
            model_prey = prey_algo.load(f"policies/model_multiprey_{prey_algo.__name__}", env_prey, tensorboard_log="./logs")
        else:
            model_prey = prey_algo("MlpPolicy", env_prey, verbose=1, tensorboard_log="./logs")

        models = {'predator':model_predator, 'prey': model_prey}
        ma_env.set_agent_models(models=models)

        total_timesteps_per_agent = 500_000
        training_iterations = 10
        steps_per_iteration = total_timesteps_per_agent // training_iterations

        for i in range(training_iterations):
            print(f"Training iteration {i}")
            for model_name, model in models.items():
                algo_name = model.__class__.__name__
                #if model_name == 'predator':
                #if model_name == 'prey':
                model.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"multi{model_name}_{algo_name}")

        ma_env.close()

        model_predator.save(f"policies/model_multipredator_{predator_algo.__name__}")
        model_prey.save(f"policies/model_multiprey_{prey_algo.__name__}")
        

    # TESTING SECTION
    ma_env = MultiPredatorPreyMAEnv(n_predators=n_predators, perimeter_side=10, max_speed_predator=1, max_speed_prey=1,render=True)
    ma_env = TimeLimitMAEnv(ma_env, max_episode_steps=250)

    model_predator = predator_algo.load(f"policies/model_multipredator_{predator_algo.__name__}")
    model_prey = prey_algo.load(f"policies/model_multiprey_{prey_algo.__name__}")

    models = {'predator':model_predator, 'prey': model_prey}
    total_rewards = {'predator':0, 'prey':0}
    n_episodes = 50
    for n_ep in range(n_episodes):
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
    
            predator_rewards = 0
            for i in range(n_predators):
                predator_rewards += rewards[f'predator_{i}']
            total_rewards['predator'] += predator_rewards/n_predators
            total_rewards['prey'] += rewards['prey']

        print("Average episode rewards:", {k: f"{v/(n_ep+1):.2f}" for k, v in total_rewards.items()})

    ma_env.close()