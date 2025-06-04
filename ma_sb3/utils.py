from ma_sb3 import BaseMAEnv, AgentInfo
import tempfile
import os
import time
from stable_baselines3 import SAC, TD3, DDPG, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor
from ma_sb3.base import MultiAgentSharedVecEnv
import random
    
class TimeLimitCallback(BaseCallback):
    def __init__(self, max_duration_seconds: int, verbose=0):
        super().__init__(verbose)
        self.max_duration_seconds = max_duration_seconds
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if time.time() - self.start_time > self.max_duration_seconds:
            if self.verbose:
                print(f"⏱️ Fin del entrenamiento tras {self.max_duration_seconds} segundos.")
            return False
        return True
    
def create_temp_model_filenames(models):
    temp_model_files = {}
    for model_name, model in models.items():
        temp_file = tempfile.NamedTemporaryFile(mode='w+b', delete=True)
        temp_model_files[model_name] = temp_file.name
    return temp_model_files

def save_temp_model(model, model_name, temp_model_filenames):
    temp_file = temp_model_filenames[model_name]
    model.save(temp_file)
    if model.__class__ in [SAC, TD3, DDPG, DQN]:
        model.save_replay_buffer(temp_file + ".pkl")

def load_temp_model(model_name, agent_env_map, model_algo_map, temp_model_filenames):
    env = agent_env_map[model_name]
    algo = model_algo_map[model_name][0]
    algo_params = model_algo_map[model_name][1]
    temp_file = temp_model_filenames[model_name]
    model = algo.load(temp_file, env=env, **algo_params)
    if algo in [SAC, TD3, DDPG, DQN]:
        model.load_replay_buffer(temp_file + ".pkl")
    return model

def load_temp_models(agent_env_map, model_algo_map, temp_model_filenames):
    models = {}
    for model_name, env in agent_env_map.items():
        algo = model_algo_map[model_name][0]
        algo_params = model_algo_map[model_name][1]
        temp_file = temp_model_filenames[model_name]
        model = algo.load(temp_file, env=env, **algo_params)
        models[model_name] = model
        if algo in [SAC, TD3, DDPG, DQN]:
            model.load_replay_buffer(temp_file + ".pkl")
    return models

def save_temp_models(models, temp_model_files):
    for model_name, model in models.items():
        temp_file = temp_model_files[model_name]
        model.save(temp_file)
        if model.__class__ in [SAC, TD3, DDPG, DQN]:
            model.save_replay_buffer(temp_file + ".pkl")

def delete_temp_model_files(temp_model_files):
    for temp_file in temp_model_files.values():
        if os.path.exists(temp_file):
            os.remove(temp_file)

def ma_train_vec(ma_env, model_algo_map, models_to_train='__all__', models_to_load=None,
            num_learning_agents = 1, 
            total_timesteps_per_model=10_000, training_iterations=2, tb_log_suffix=""):        

        all_agents = list(ma_env.agents.keys())

        # Randomly select some of them for training
        learning_agents = random.sample(all_agents, num_learning_agents)

        # Wrap in VecEnv only the learning agents
        vec_env = MultiAgentSharedVecEnv(ma_env, learning_agents)
        vec_env = VecMonitor(vec_env)

        # Train the agents
        return ma_train(ma_env=vec_env, model_algo_map=model_algo_map, models_to_train=models_to_train,
             models_to_load=models_to_load, total_timesteps_per_model=total_timesteps_per_model,
             training_iterations=training_iterations, tb_log_suffix=tb_log_suffix)


def ma_train(ma_env, model_algo_map, models_to_train='__all__', models_to_load=None,
             total_timesteps_per_model=10_000, training_iterations=2, tb_log_suffix=""):        
        """
        Trains multiple agents in a multi-agent environment using different models and algorithms.
        Args:
            ma_env (MultiAgentEnv): The multi-agent environment.
            model_algo_map (dict): A dictionary mapping agent names to tuples of (algorithm, algorithm_params).
            models_to_train (list or str, optional): The list of model names to train. Defaults to '__all__'.
            models_to_load (dict, optional): A dictionary mapping model names to pre-trained models to load based on their path. Defaults to None.
            total_timesteps_per_model (int, optional): The total number of timesteps to train each model. Defaults to 10_000.
            training_iterations (int, optional): The number of training iterations. Defaults to 2.
            tb_log_suffix (str, optional): The suffix to append to the TensorBoard log name. Defaults to "".
        Returns:
            dict: A dictionary mapping agent names to trained models.
        """

        agent_env_map = {agent_info.model_name: ma_env for agent_info in ma_env.agents.values()}

        # Create agent models, either from scratch or load pre-trained models
        models = {}
        for model_name, env in agent_env_map.items():
            algo = model_algo_map[model_name][0]
            algo_params = model_algo_map[model_name][1]
            if models_to_load is not None and model_name in models_to_load:
                model = algo.load(models_to_load[model_name], env=env, **algo_params)
                if algo in [SAC, TD3, DDPG, DQN]:
                    model.load_replay_buffer(models_to_load[model_name] + ".pkl")
            else:
                model = algo(env=env, **algo_params)
            models[model_name] = model

        previous_model_filenames = create_temp_model_filenames(models)
        save_temp_models(models, previous_model_filenames)

        trained_model_filenames = create_temp_model_filenames(models)

        ma_env.set_agent_models(models=models)

        steps_per_iteration = total_timesteps_per_model // training_iterations
       
        if models_to_load is None or len(models_to_load) == 0:
            reset_timesteps = True
        else:
            reset_timesteps = False

        if models_to_train == '__all__':
            models_to_train = list(models.keys())


        for i in range(training_iterations):
            print(f"Training iteration {i+1} of {training_iterations}...")
            for model_name, model in models.items():
                if model_name in models_to_train:
                    algo_name = model.__class__.__name__
                    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./checkpoints/{model_name}_{algo_name}_{tb_log_suffix}")
                    time_limit_callback = TimeLimitCallback(max_duration_seconds=60*120)  # 2 hour limit
                    #callback = CallbackList([time_limit_callback, checkpoint_callback])
                    callback = CallbackList([checkpoint_callback])
                    print(f"Training {model_name} with {algo_name}...")
                    model.learn(total_timesteps=steps_per_iteration, callback=callback, progress_bar=True, reset_num_timesteps=reset_timesteps, tb_log_name=f"{model_name}_{algo_name}_{tb_log_suffix}")
                    
                # Save the model (trained or not) to a temporary file
                save_temp_model(model, model_name, trained_model_filenames)
                # Reload the previous model
                models[model_name] = load_temp_model(model_name, agent_env_map, model_algo_map, previous_model_filenames)
                # Reassign the models
                ma_env.set_agent_models(models=models)

            reset_timesteps = False # To avoid resetting the timesteps after the first iteration
            
            # Load the models (trained or not) back into the environment
            models = load_temp_models(agent_env_map, model_algo_map, trained_model_filenames)
            ma_env.set_agent_models(models=models)
            # Save the models as the previous models
            save_temp_models(models, previous_model_filenames)

        delete_temp_model_files(previous_model_filenames)
        delete_temp_model_files(trained_model_filenames)

        return models

def ma_evaluate(ma_env, models, total_episodes=100, verbose=False):
    """
    Evaluates the performance of multiple agents in a multi-agent environment using the given models.
    Args:
        ma_env (MultiAgentEnv): The multi-agent environment to evaluate the agents in.
        models (dict): A dictionary mapping model names to their corresponding models.
        total_episodes (int, optional): The total number of episodes to run the evaluation. Defaults to 100.
    Returns:
        tuple: A tuple containing two dictionaries:
            - average_reward_agent: A dictionary mapping agent ids to their average rewards across all episodes.
            - average_reward_per_model: A dictionary mapping model names to their average rewards across all episodes.
    """
    list_episodes_rewards = {agent_id: [] for agent_id in ma_env.agents.keys()}
    agentid_modelname_map = {agent_id: agent_info.model_name for agent_id, agent_info in ma_env.agents.items()}

    for _ in range(total_episodes):
        obs, info = ma_env.reset()
        terminated = False
        truncated = False
        episode_rewards = {agent_id: 0 for agent_id in obs.keys()}

        while not terminated and not truncated:
            actions = {}
            for agent_id, agent_obs in obs.items():
                model_id = agentid_modelname_map[agent_id]
                actions[agent_id] = models[model_id].predict(agent_obs)[0]
            obs, rewards, terminated, truncated, infos = ma_env.step_all(actions)
            # Update current episode rewards for each agent
            episode_rewards = {agent_id: episode_rewards[agent_id] + rewards[agent_id] for agent_id in rewards.keys()}

        if terminated or truncated:
            # Add the episode rewards to the list of rewards for each agent
            for agent_id, reward in episode_rewards.items():
                list_episodes_rewards[agent_id].append(reward)
            if verbose:
                print(f"Episode {_+1} completed. Rewards: {episode_rewards}")
    
    # Calculate average reward per agent
    average_reward_agent = {agent_id: sum(rewards)/total_episodes for agent_id, rewards in list_episodes_rewards.items()}
    
    # Calculate average reward per model
    average_reward_per_model = {model_name: 0 for model_name in set(agentid_modelname_map.values())}
    # First aggregate the rewards per model
    for agent_id, model_name in agentid_modelname_map.items():
        average_reward_per_model[model_name] += average_reward_agent[agent_id]
    # Then divide by the number of agents per model
    for model_name in average_reward_per_model.keys():
        average_reward_per_model[model_name] /= len([agent_id for agent_id in agentid_modelname_map if agentid_modelname_map[agent_id] == model_name])

    return average_reward_agent, average_reward_per_model
