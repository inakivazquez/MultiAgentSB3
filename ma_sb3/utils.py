from ma_sb3 import BaseMAEnv, AgentMAEnv
from copy import deepcopy
    

def ma_train(ma_env, model_algo_map, models_to_train='all', models_to_load={},
             total_timesteps_per_model=10_000, training_iterations=2, tb_log_suffix=""):        
        """
        Trains multiple agents in a multi-agent environment using different models and algorithms.
        Args:
            ma_env (MultiAgentEnv): The multi-agent environment.
            model_algo_map (dict): A dictionary mapping agent names to tuples of (algorithm, algorithm_params).
            models_to_train (list or str, optional): The list of model names to train. Defaults to 'all'.
            models_to_load (dict, optional): A dictionary mapping model names to pre-trained models to load based on their path. Defaults to {}.
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

        steps_per_iteration = total_timesteps_per_model // training_iterations

        if models_to_train == 'all':
            models_to_train = list(models.keys())

        for i in range(training_iterations):
            print(f"Training iteration {i} of {training_iterations}...")
            for model_name, model in models.items():
                if model_name in models_to_train:
                    algo_name = model.__class__.__name__
                    print(f"Training {model_name} with {algo_name}...")
                    model.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{model_name}_{algo_name}_{tb_log_suffix}")
        return models


def ma_train2(ma_env, model_algo_map, models_to_train='all', models_to_load={},
             total_timesteps_per_model=10_000, training_iterations=2, tb_log_suffix=""):        
        """
        Trains multiple agents in a multi-agent environment using different models and algorithms.
        Args:
            ma_env (MultiAgentEnv): The multi-agent environment.
            model_algo_map (dict): A dictionary mapping agent names to tuples of (algorithm, algorithm_params).
            models_to_train (list or str, optional): The list of model names to train. Defaults to 'all'.
            models_to_load (dict, optional): A dictionary mapping model names to pre-trained models to load based on their path. Defaults to {}.
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

        previous_models = deepcopy(models)
        ma_env.set_agent_models(models=previous_models)

        steps_per_iteration = total_timesteps_per_model // training_iterations

        if models_to_train == 'all':
            models_to_train = list(models.keys())

        for i in range(training_iterations):
            print(f"Training iteration {i} of {training_iterations}...")
            for model_name, model in models.items():
                if model_name in models_to_train:
                    algo_name = model.__class__.__name__
                    print(f"Training {model_name} with {algo_name}...")
                    model.learn(total_timesteps=steps_per_iteration, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"{model_name}_{algo_name}_{tb_log_suffix}")
                    ma_env.set_agent_models(models=previous_models)

            previous_models = deepcopy(models)
            ma_env.set_agent_models(models=previous_models)

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
