from stable_baselines3.common.vec_env import VecEnv
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import random

class AgentInfo():
    def __init__(self, agent_id, model_name, observation_space, action_space) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.model_name = model_name
        self.model = None

        self.observation_space = observation_space
        self.action_space = action_space

    def set_model(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        action = self.model.predict(obs, deterministic=deterministic)
        return action


class BaseMAEnv():

    def __init__(self):
        self.agents = {}
        self.previous_observation = {}

    def register_agent(self, agent_id, observation_space, action_space, model_name=None):
        if model_name is None:
            model_name = agent_id
        # Create the agent
        agent_info = AgentInfo(agent_id, model_name, observation_space, action_space)
        # Add the agent and model name to the dictionary
        self.agents[agent_id] = agent_info

    def set_agent_models(self, models):
        """
        Sets the models for all agents.

        Parameters:
        - models (dict): A dictionary where keys are the model identifiers and values are the models to be set.
        """
        for agent_id, agent_info in self.agents.items():
            model_name = agent_info.model_name
            if model_name in models:
                agent_info.set_model(models[model_name])
            else:
                raise ValueError(f"Model for agent_id '{agent_id}' is not provided in the models dictionary.")

    def step_all(self, agent_actions):
        # Execute steps concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.step_agent, agent_id, action) for agent_id, action in agent_actions.items()]
            for future in futures:
                future.result()  # Wait for all futures to complete
        
        self.sync_wait_for_actions_completion()

        obs, rewards, terminated, truncated, info = self.get_state()

        return obs, rewards, terminated, truncated, info

    def get_state(self):
        # Evaluate the state of the environment, part of this informaton may be used to get observations
        rewards, terminated, truncated, infos = self.evaluate_env_state()

        # Get the observations for all agents
        obs = {}
        for agent_id in self.agents.keys():
            obs[agent_id] = self.get_observation(agent_id)

        self.previous_observation = obs # Required for the multi-agent step process, we need to reuse the previous observation
        return obs, rewards, terminated, truncated, infos

    def close(self):
        # Closes the environment
        pass

    # Functions to be implemented by the subclass

    def step_agent(self, agent_id, action):
        # Must be implemented in the subclass
        # Executes the action for the agent_id
        raise NotImplementedError
    
    def get_observation(self, agent_id):
        # Must be implemented in the subclass
        # Returns the observation for the agent_id
        raise NotImplementedError

    def sync_wait_for_actions_completion(self):
        # Must be implemented, even if it is just a 'pass' if no synchronization is needed
        raise NotImplementedError

    def evaluate_env_state(self):
        # For the current state of the environment, it must return rewards, terminated, truncated, infos
        raise NotImplementedError

    def reset(self, seed=None):
        # Must return obs, info
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        

class TimeLimitMAEnv:
    
    def __init__(self, env:BaseMAEnv, max_episode_steps=None) -> None:
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

    def __getattr__(self, name):
        # Delegate attribute and method access to self.env
        return getattr(self.env, name)
    
    """@property
    def agents(self):
        return self.env.agents
        
    @property
    def previous_observation(self):
        return self.env.previous_observation"""
        
    def step_all(self, agent_actions):
        obs, rewards, terminated, truncated, info = self.env.step_all(agent_actions)
        self.current_step += 1

        if self.max_episode_steps is not None and self.current_step >= self.max_episode_steps:
            truncated = True

        return obs, rewards, terminated, truncated, info

    def reset(self, seed=None):
        self.current_step = 0
        return self.env.reset(seed)

    """def register_agent(self, agent_id, model_name, observation_space, action_space):
        return self.env.register_agent(agent_id, model_name, observation_space, action_space)

    def set_agent_models(self, models):
        return self.env.set_agent_models(models)

    def get_state(self):
        return self.env.get_state()

    def close(self):
        return self.env.close()

    def step_agent(self, agent_id, action):
        return self.env.step_agent(agent_id, action)
    
    def get_observation(self, agent_id):
        return self.env.get_observation(agent_id)

    def sync_wait_for_actions_completion(self):
        return self.env.sync_wait_for_actions_completion()

    def evaluate_env_state(self):
        return self.env.evaluate_env_state()"""



class MultiAgentSharedVecEnv(VecEnv):
    def __init__(self, shared_env, learning_agent_ids):
        self.shared_env = shared_env
        self.learning_agent_ids = learning_agent_ids
        self.num_learning_agents = len(learning_agent_ids)

        # Assume homogeneous spaces, take the first agent's spaces
        sample_agent = shared_env.agents[learning_agent_ids[0]]
        observation_space = sample_agent.observation_space
        action_space = sample_agent.action_space

        super().__init__(
            num_envs=self.num_learning_agents,
            observation_space=observation_space,
            action_space=action_space
        )

        self.actions = None

    @property
    def agents(self):
        return self.shared_env.agents
        
    """def get_agents_envs(self):
        return self.shared_env.get_agents_envs()"""

    def set_agent_models(self, models):
        return self.shared_env.set_agent_models(models)

    def reset(self, seed=None, options=None):
        obs, info = self.shared_env.reset(seed=seed)
        obs_arr = [obs[agent_id] for agent_id in self.learning_agent_ids]
        return np.array(obs_arr)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        agent_actions = {}

        # Collect actions for learning agents (from algorithm)
        for agent_id, action in zip(self.learning_agent_ids, self.actions):
            agent_actions[agent_id] = action

        # Predict actions for the other agents
        for agent_id, agent_info in self.shared_env.agents.items():
            if agent_id not in self.learning_agent_ids:
                obs = self.shared_env.previous_observation[agent_id]
                agent_actions[agent_id] = agent_info.predict(obs)[0]

        # Step all agents together
        obs, rewards, terminated, truncated, infos = self.shared_env.step_all(agent_actions)
        episode_done = terminated or truncated

        obs_arr, rewards_arr, dones_arr, infos_arr = [], [], [], []

        # First store the current observations (before reset)
        for agent_id in self.learning_agent_ids:
            obs_arr.append(obs[agent_id])
            rewards_arr.append(rewards[agent_id])
            dones_arr.append(episode_done)

            info = {}
            if episode_done:
                info["terminal_observation"] = obs[agent_id]
            infos_arr.append(info)

        # After terminal observation is recorded, reset and update obs
        if episode_done:
            new_obs, _ = self.shared_env.reset()
            obs_arr = [new_obs[agent_id] for agent_id in self.learning_agent_ids]

        return (
            np.array(obs_arr),
            np.array(rewards_arr, dtype=np.float32),
            np.array(dones_arr, dtype=np.bool_),
            infos_arr,
        )



    def close(self):
        self.shared_env.close()

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None] * self.num_envs
        raise NotImplementedError(f"get_attr('{attr_name}') is not supported.")

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("set_attr is not supported for this wrapper.")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("env_method is not supported for this wrapper.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
    

