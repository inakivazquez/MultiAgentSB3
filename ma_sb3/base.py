from concurrent.futures import ThreadPoolExecutor
from gymnasium import Env
class MAAgentEnv(Env):
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
            return self.model.predict(obs)


class BaseSharedEnv():

    agents = {}
    previous_observation = {}

    def register_agent(self, agent_id, observation_space, action_space):
        # Create the agent environment
        agent_env = MAAgentEnv(self, agent_id, observation_space, action_space)
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
        def step_agent(agent_id, action):
            step_function = getattr(self, f'step_agent_{agent_id}')
            step_function(action)

        # Execute steps concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(step_agent, agent_id, action) for agent_id, action in agent_actions.items()]
            for future in futures:
                future.result()  # Wait for all futures to complete
        
        self.wait_for_actions_completion()

        obs, rewards, terminated, truncated, info = self.get_full_state()

        return obs, rewards, terminated, truncated, info
                
    def wait_for_actions_completion(self):
        # Must be implemented, even if it is just a pass
        raise NotImplementedError

    def predict_other_agents_actions(self, agent_id):
        if agent_id not in self.agents:
            raise ValueError("Invalid agent_id")
        other_agents_predictions = {}
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                # Predict the action of the other agent using its environment's predict method
                other_agents_predictions[other_agent_id] = self.agents[other_agent_id].predict(self.previous_observation[other_agent_id])[0][0]
        return other_agents_predictions

    def get_full_state(self):
         obs, rewards, terminated, truncated, infos = self.get_env_full_state()
         self.previous_observation = obs # Required for the multi-agent step process, we need to reuse the previous observation
         return obs, rewards, terminated, truncated, infos

    def get_env_full_state(self):
        # This is the one to be implemented by the subclass
        raise NotImplementedError

    def reset(self, seed=0):
        # Reset the environment
        raise NotImplementedError