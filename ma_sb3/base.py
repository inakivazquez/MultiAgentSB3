from concurrent.futures import ThreadPoolExecutor
from gymnasium import Env
class AgentMAEnv(Env):
        def __init__(self, shared_env, agent_id, observation_space, action_space) -> None:
            super().__init__()
            self.shared_env = shared_env
            self.agent_id = agent_id
    
            self.observation_space = observation_space
            self.action_space = action_space
    
        def step(self, actions):
            all_actions = self.shared_env.predict_other_agents_actions(self.agent_id)
            all_actions[self.agent_id] = actions 
            all_observations, rewards, terminated, truncated, info = self.shared_env.step_all(all_actions)
            observations = all_observations[self.agent_id]
            reward = rewards[self.agent_id]

            return observations, reward, terminated, truncated, info
    
        def reset(self, seed=0):
            obs, info = self.shared_env.reset(seed)
            return obs[self.agent_id], info

        def set_model(self, model):
            self.model = model

        def predict(self, obs):
            action = self.model.predict(obs)
            return action


class BaseMAEnv():

    agents = {}
    previous_observation = {}

    def register_agent(self, agent_id, observation_space, action_space):
        # Create the agent environment
        agent_env = AgentMAEnv(self, agent_id, observation_space, action_space)
        # Add the agent environment to the dictionary
        self.agents[agent_id] = agent_env

    def get_agents_envs(self):
        # Return the agents
        return self.agents

    def set_agent_models(self, models):
        """
        Sets the models for all agents.

        Parameters:
        - models (dict): A dictionary where keys are the agent identifiers and values are the models to be set.
        """
        for agent_id, agent in self.agents.items():
            if agent_id in models:
                agent.set_model(models[agent_id])
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
                
    def predict_other_agents_actions(self, agent_id):
        if agent_id not in self.agents:
            raise ValueError("Invalid agent_id")
        other_agents_predictions = {}
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                # Predict the action of the other agent using its environment's predict method
                other_agents_predictions[other_agent_id] = self.agents[other_agent_id].predict(self.previous_observation[other_agent_id])[0]
        return other_agents_predictions

    def get_state(self):
         obs = {}
         for agent_id in self.agents.keys():
             obs[agent_id] = self.get_observation(agent_id)
         rewards, terminated, truncated, infos = self.get_env_state_results()
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

    def get_env_state_results(self):
        # For the current state of the environment, it must return rewards, terminated, truncated, infos
        raise NotImplementedError

    def reset(self, seed=0):
        # Must return obs, info
        raise NotImplementedError
        

class TimeLimitMAEnv(BaseMAEnv):

    max_episode_steps = None
    current_step = 0
    
    def __init__(self, env, max_episode_steps=None) -> None:
        self.env = env
        self.max_episode_steps = max_episode_steps
        for agent_id, agent_env in self.agents.items():
            agent_env.shared_env = self

    def step_all(self, agent_actions):
        obs, rewards, terminated, truncated, info = self.env.step_all(agent_actions)
        self.current_step += 1

        if self.max_episode_steps is not None and self.current_step >= self.max_episode_steps:
            truncated = True

        return obs, rewards, terminated, truncated, info

    def reset(self, seed=0):
        self.current_step = 0
        return self.env.reset(seed)

    def register_agent(self, agent_id, observation_space, action_space):
        return self.env.register_agent(agent_id, observation_space, action_space)

    def set_agent_models(self, models):
        return self.env.set_agent_models(models)
                
    def sync_wait_for_actions_completion(self):
        return self.env.wait_for_actions_completion()

    def predict_other_agents_actions(self, agent_id):
        return self.env.predict_other_agents_actions(agent_id)

    def get_state(self):
        return self.env.get_full_state()

    def get_env_state_results(self):
        return self.env.get_env_full_state()

    def close(self):
        return self.env.close()
