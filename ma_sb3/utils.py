from ma_sb3 import BaseMAEnv, AgentMAEnv
import tempfile
import os
import time

    

def ma_train(ma_env, model_algo_map, models_to_train='all', models_to_load=None,
             total_timesteps_per_model=10_000, training_iterations=2, tb_log_suffix=""):        
        """
        Trains multiple agents in a multi-agent environment using different models and algorithms.
        Args:
            ma_env (MultiAgentEnv): The multi-agent environment.
            model_algo_map (dict): A dictionary mapping agent names to tuples of (algorithm, algorithm_params).
            models_to_train (list or str, optional): The list of model names to train. Defaults to 'all'.
            models_to_load (dict, optional): A dictionary mapping model names to pre-trained models to load based on their path. Defaults to None.
            total_timesteps_per_model (int, optional): The total number of timesteps to train each model. Defaults to 10_000.
            training_iterations (int, optional): The number of training iterations. Defaults to 2.
            tb_log_suffix (str, optional): The suffix to append to the TensorBoard log name. Defaults to "".
        Returns:
            dict: A dictionary mapping agent names to trained models.
        """

        agent_env_map = {model_name: env for env, model_name in ma_env.agents.values()}

        # Create agent models, either from scratch or load pre-trained models
        models = {}
        for model_name, env in agent_env_map.items():
            algo = model_algo_map[model_name][0]
            algo_params = model_algo_map[model_name][1]
            if models_to_load is not None and model_name in models_to_load:
                model = algo.load(models_to_load[model_name], env=env, **algo_params)
            else:
                model = algo(env=env, **algo_params)
            models[model_name] = model

        ma_env.set_agent_models(models=models)

        if models_to_load is None or len(models_to_load) == 0:
            reset_timesteps = True
        else:
            reset_timesteps = False

        steps_per_iteration = total_timesteps_per_model // training_iterations

        if models_to_train == 'all':
            models_to_train = list(models.keys())

        for i in range(training_iterations):
            print(f"Training iteration {i+1} of {training_iterations}...")
            for model_name, model in models.items():
                if model_name in models_to_train:
                    algo_name = model.__class__.__name__
                    print(f"Training {model_name} with {algo_name}...")
                    model.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=reset_timesteps, tb_log_name=f"{model_name}_{algo_name}_{tb_log_suffix}")
            reset_timesteps = False # To avoid resetting the timesteps after the first iteration
        return models

def create_temp_model_filenames(models):
    temp_model_files = {}
    for model_name, model in models.items():
        temp_file = tempfile.NamedTemporaryFile(mode='w+b', delete=True)
        temp_model_files[model_name] = temp_file.name
    return temp_model_files

def save_temp_model(model, model_name, temp_model_filenames):
    temp_file = temp_model_filenames[model_name]
    model.save(temp_file)

def load_temp_model(model_name, agent_env_map, model_algo_map, temp_model_filenames):
    env = agent_env_map[model_name]
    algo = model_algo_map[model_name][0]
    algo_params = model_algo_map[model_name][1]
    temp_file = temp_model_filenames[model_name]
    model = algo.load(temp_file, env=env, **algo_params)
    return model

def load_temp_models(agent_env_map, model_algo_map, temp_model_filenames):
    models = {}
    for model_name, env in agent_env_map.items():
        algo = model_algo_map[model_name][0]
        algo_params = model_algo_map[model_name][1]
        temp_file = temp_model_filenames[model_name]
        model = algo.load(temp_file, env=env, **algo_params)
        models[model_name] = model
    return models

def save_temp_models(models, temp_model_files):
    for model_name, model in models.items():
        temp_file = temp_model_files[model_name]
        model.save(temp_file)

def delete_temp_model_files(temp_model_files):
    for temp_file in temp_model_files.values():
        if os.path.exists(temp_file):
            os.remove(temp_file)

def ma_train2(ma_env, model_algo_map, models_to_train='all', models_to_load=None,
             total_timesteps_per_model=10_000, training_iterations=2, tb_log_suffix=""):        
        """
        Trains multiple agents in a multi-agent environment using different models and algorithms.
        Args:
            ma_env (MultiAgentEnv): The multi-agent environment.
            model_algo_map (dict): A dictionary mapping agent names to tuples of (algorithm, algorithm_params).
            models_to_train (list or str, optional): The list of model names to train. Defaults to 'all'.
            models_to_load (dict, optional): A dictionary mapping model names to pre-trained models to load based on their path. Defaults to None.
            total_timesteps_per_model (int, optional): The total number of timesteps to train each model. Defaults to 10_000.
            training_iterations (int, optional): The number of training iterations. Defaults to 2.
            tb_log_suffix (str, optional): The suffix to append to the TensorBoard log name. Defaults to "".
        Returns:
            dict: A dictionary mapping agent names to trained models.
        """

        agent_env_map = {model_name: env for env, model_name in ma_env.agents.values()}

        # Create agent models, either from scratch or load pre-trained models
        models = {}
        for model_name, env in agent_env_map.items():
            algo = model_algo_map[model_name][0]
            algo_params = model_algo_map[model_name][1]
            if models_to_load is not None and model_name in models_to_load:
                model = algo.load(models_to_load[model_name], env=env, **algo_params)
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

        if models_to_train == 'all':
            models_to_train = list(models.keys())

        for i in range(training_iterations):
            print(f"Training iteration {i+1} of {training_iterations}...")
            for model_name, model in models.items():
                if model_name in models_to_train:
                    algo_name = model.__class__.__name__
                    print(f"Training {model_name} with {algo_name}...")
                    model.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=reset_timesteps, tb_log_name=f"{model_name}_{algo_name}_{tb_log_suffix}")
                    
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


def ma_evaluate(ma_env, models, total_episodes=100):
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
    agentid_modelname_map = {agent_id: env_model_map[1] for agent_id, env_model_map in ma_env.agents.items()}

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
