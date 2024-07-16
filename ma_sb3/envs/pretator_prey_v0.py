from concurrent.futures import ThreadPoolExecutor
from ma_sb3 import MAAgentEnv, BaseSharedEnv

from gymnasium.spaces import Box
import numpy as np

import pybullet as p
import pybullet_data
import time

import pybullet as p
import random

class PredatorPreyEnv(BaseSharedEnv):

    SIMULATION_STEP_DELAY = 1. / 240.

    def __init__(self, render=False):
        super(PredatorPreyEnv, self).__init__()
        self.render_mode = render
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        #p.setTimeStep(1e-3)
        p.setRealTimeSimulation(0)
        # Hide debug panels in PyBullet
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        # Reorient the debug camera
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[0,-1,0])


        # Load the plane and objects
        self.plane_id = p.loadURDF("plane.urdf", [0,0,0])
        self.predator_id = p.loadURDF("cube.urdf", [0, 0, 0], useFixedBase=False, globalScaling=0.3,)
        p.changeVisualShape(self.predator_id, -1, rgbaColor=[0.8, 0.1, 0.1, 1])
        self.prey_id = p.loadURDF("cube.urdf", [1, 1, 0], useFixedBase=False, globalScaling=0.2)
        #p.changeDynamics(self.predator_id, -1, lateralFriction=1)
        #p.changeDynamics(self.prey_id, -1, lateralFriction=1)

        self.draw_perimeter(6)

        # Create the agents
        self.predator_observation_space = Box(low=np.array([-5, -5]), high=np.array([5, 5]), shape=(2,), dtype=np.float32)
        self.prey_observation_space = Box(low=np.array([-5, -5]), high=np.array([5, 5]), shape=(2,), dtype=np.float32)

        self.predator_action_space = Box(low=np.array([-10, -10]), high=np.array([10, 10]), shape=(2,), dtype=np.float32)
        self.prey_action_space = Box(low=np.array([-12, -12]), high=np.array([12, 12]), shape=(2,), dtype=np.float32)

        # Create the agent environments
        self.agent_predator_env = MAAgentEnv(self, "predator", self.predator_observation_space, self.predator_action_space)
        self.agent_prey_env = MAAgentEnv(self, "prey", self.prey_observation_space, self.prey_action_space)

        # Add the agents to the dictionary for further management
        self.agents = {
            "predator": self.agent_predator_env,
            "prey": self.agent_prey_env
        }


    def step_simulation(self):
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def wait_for_actions_completion(self, sim_steps=10):
        for _ in range(sim_steps):
            self.step_simulation()
        
    def reset(self, seed=0):
        limit_spawn_perimeter = 2
        random_coor = lambda: random.uniform(-limit_spawn_perimeter, limit_spawn_perimeter)
        p.resetBasePositionAndOrientation(self.predator_id, [random_coor(), random_coor(), 0.5], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.prey_id, [random_coor(), random_coor(), 0.5], [0, 0, 0, 1])
        p.resetBaseVelocity(self.predator_id, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.prey_id, [0, 0, 0], [0, 0, 0])
        self.wait_for_actions_completion(100)

        obs, _, _, _, info = self.get_full_state()
        return obs, info

    def step_agent_predator(self, action):
        force_x = action[0]
        force_y = action[1]
        self.move(self.predator_id, force_x, force_y)

    def step_agent_prey(self, action):
        force_x = action[0]
        force_y = action[1]
        self.move(self.prey_id, force_x, force_y)

    def get_env_full_state(self):
        obs = {
            "predator": self.get_observations_predator(),
            "prey": self.get_observations_prey()
        }

        truncated = False
        rewards = {'predator': 0, 'prey': 0}
        infos = {}   

        predator_pos = self.get_position_predator()
        prey_pos = self.get_position_prey()
        #distance = np.linalg.norm(predator_pos - prey_pos)  # Calculate distance between the two, not used in this example
        if(p.getContactPoints(self.predator_id, self.prey_id)):
            # Predator wins
            terminated = True
            rewards['predator'] = 10
            rewards['prey'] = -10
            print("Touched!")
        else:        
            limit_perimeter = 3
            predator_out = np.any(np.logical_or(predator_pos < -limit_perimeter, predator_pos > limit_perimeter))
            prey_out = np.any(np.logical_or(prey_pos < -limit_perimeter, prey_pos > limit_perimeter))
            if predator_out:
                terminated = True
                rewards['predator'] = -100
                print("Predator: Out of bounds!")
                rewards['prey'] = 10 # Prey wins
            elif prey_out:
                terminated = True
                rewards['prey'] = -100
                print("Prey: Out of bounds!")
                rewards['predator'] = 10 # Predator wins
            else:
                # prey wins +1 per time step, predator loses 1
                rewards['predator'] -= 0.1
                rewards['prey'] += 0.1
                terminated = False

        return obs, rewards, terminated, truncated, infos

    def get_observations_predator(self):
        predator_pos = self.get_position_predator()
        prey_pos = self.get_position_prey()
        return prey_pos-predator_pos

    def get_observations_prey(self):
        predator_pos = self.get_position_predator()
        prey_pos = self.get_position_prey()
        return predator_pos-prey_pos
    
    def get_position_predator(self):
        predator_pos,_ = p.getBasePositionAndOrientation(self.predator_id)
        return np.array([predator_pos[:2]])

    def get_position_prey(self):
        prey_pos,_ = p.getBasePositionAndOrientation(self.prey_id)
        return np.array([prey_pos[:2]])

    def move(self, agent_id, force_x, force_y):
        factor = 100
        force_x *= factor
        force_y *= factor
        p.applyExternalForce(agent_id, -1, [force_x, force_y, 0], [0, 0, 0], p.LINK_FRAME)

    def render(self, mode='human'):
        pass  # Rendering handled in real-time if GUI mode is enabled

    def close(self):
        p.disconnect()

    def draw_perimeter(self, side_length):
        height = 0.4
        half_size = side_length / 2.0
        half_height = height / 2.0
        color = [0, 0.4, 0, 0.8]  # Green color

        # Create visual shape for the box
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size, half_height],
            rgbaColor=color,
            specularColor=[0, 0, 0]
        )

        # Create collision shape for the box
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size, half_height]
        )

        # Create the multi-body object
        perimeter_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, half_height]
        )

        p.changeDynamics(perimeter_id, -1, lateralFriction=1)

