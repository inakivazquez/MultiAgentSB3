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

    def get_agents_envs(self):
        # Create the agents
        raise NotImplementedError

    def step_all(self, agent_actions):
        def step_agent(agent_id, action):
            step_function = getattr(self, f'step_agent_{agent_id}')
            step_function(action)

        # Execute steps concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(step_agent, agent_id, action) for agent_id, action in agent_actions.items()]
            for future in futures:
                future.result()  # Wait for all futures to complete
        
        self.wait_actions_completion()

        obs, rewards, terminated, truncated, info = self.get_full_state()

        return obs, rewards, terminated, truncated, info
                
    def wait_actions_completion(self):
        pass

    def get_full_state(self):
        # Get the observations for all agents
        raise NotImplementedError

    def reset(self, seed=0):
        # Reset the environment
        raise NotImplementedError