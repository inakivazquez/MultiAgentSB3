from ma_sb3.envs import PredatorPreyEnv

from gymnasium.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO
import pybullet as p


# Example of running the environment
if __name__ == "__main__":

    train = True
    train = False

    if train:
        shared_environment = PredatorPreyEnv(render=False)

        agents_envs = shared_environment.get_agents_envs()
        
        env1 = TimeLimit(agents_envs['predator'], max_episode_steps=100)
        env2 = TimeLimit(agents_envs['prey'], max_episode_steps=100)

        # Start new training
        #model_predator = PPO("MlpPolicy", env1, verbose=1, tensorboard_log="./logs")
        #model_prey = PPO("MlpPolicy", env2, verbose=1, tensorboard_log="./logs")

        # Continue training
        model_predator = PPO.load("model_predator", env1, tensorboard_log="./logs")
        model_prey = PPO.load("model_prey", env2, tensorboard_log="./logs")


        shared_environment.set_agent_models(models = {'predator':model_predator, 'prey': model_prey})

        total_timesteps = 200_000
        iterations = 50
        steps_per_iteration = total_timesteps // iterations

        for i in range(iterations):
            print(f"Training iteration {i}")
            #model_predator.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name="predator")
            model_prey.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name="prey")

        shared_environment.close()

        model_predator.save("model_predator")
        model_prey.save("model_prey")
    

    model_predator = PPO.load("model_predator")
    model_prey = PPO.load("model_prey")
    
    shared_environment = PredatorPreyEnv(render=True)
    agents_envs = shared_environment.get_agents_envs()

    for _ in range(50):
        obs, info = shared_environment.reset()
        terminated = False

        while not terminated:
            actions = {
                "predator": model_predator.predict(obs['predator'])[0][0],
                "prey": model_prey.predict(obs['prey'])[0][0]
            }

            #print("Actions:", actions)
            obs, rewards, terminated, _ , _= shared_environment.step_all(actions)

    shared_environment.close()