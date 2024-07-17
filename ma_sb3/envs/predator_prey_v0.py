from concurrent.futures import ThreadPoolExecutor
from ma_sb3 import AgentMAEnv, BaseMAEnv

from gymnasium.spaces import Box
import numpy as np

import pybullet as p
import pybullet_data
import time

import pybullet as p
import random

class PredatorPreyMAEnv(BaseMAEnv):

    SIMULATION_STEP_DELAY = 1. / 240.
    MAX_PREDATOR_SPEED = 3
    MAX_PREY_SPEED = 5

    def __init__(self, render=False):
        super(PredatorPreyMAEnv, self).__init__()
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
        self.pybullet_plane_id = p.loadURDF("plane.urdf", [0,0,0])
        self.pybullet_predator_id = p.loadURDF("cube.urdf", [0, 0, 0], useFixedBase=False, globalScaling=0.3)
        p.changeVisualShape(self.pybullet_predator_id, -1, rgbaColor=[0.8, 0.1, 0.1, 1])
        self.pybullet_prey_id = p.loadURDF("cube.urdf", [1, 1, 0], useFixedBase=False, globalScaling=0.2)
        #p.changeDynamics(self.predator_id, -1, lateralFriction=1)
        #p.changeDynamics(self.prey_id, -1, lateralFriction=1)

        self.draw_perimeter(6)

        # Create the agents
        self.register_agent(agent_id='predator',
                        observation_space=Box(low=np.array([-5, -5]), high=np.array([5, 5]), shape=(2,), dtype=np.float32),
                        action_space=Box(low=np.array([-10, -10]), high=np.array([10, 10]), shape=(2,), dtype=np.float32)
                        )
        self.register_agent(agent_id='prey',
                        observation_space=Box(low=np.array([-5, -5]), high=np.array([5, 5]), shape=(2,), dtype=np.float32),
                        action_space=Box(low=np.array([-10, -10]), high=np.array([10, 10]), shape=(2,), dtype=np.float32)
                        )

    def step_simulation(self):
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def sync_wait_for_actions_completion(self, sim_steps=10):
        for _ in range(sim_steps):
            self.step_simulation()
        
    def reset(self, seed=0):
        limit_spawn_perimeter = 2
        random_coor = lambda: random.uniform(-limit_spawn_perimeter, limit_spawn_perimeter)
        p.resetBasePositionAndOrientation(self.pybullet_predator_id, [random_coor(), random_coor(), 0.5], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.pybullet_prey_id, [random_coor(), random_coor(), 0.5], [0, 0, 0, 1])
        p.resetBaseVelocity(self.pybullet_predator_id, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.pybullet_prey_id, [0, 0, 0], [0, 0, 0])
        self.sync_wait_for_actions_completion(100)

        obs, _, _, _, info = self.get_state()
        return obs, info

    def step_agent(self, agent_id, action):
        force_x = action[0]
        force_y = action[1]

        if agent_id == 'predator':
            pybullet_object_id = self.pybullet_predator_id
            max = self.MAX_PREDATOR_SPEED
        else:
            pybullet_object_id = self.pybullet_prey_id
            max = self.MAX_PREY_SPEED

        # Limit the speed
        velocity, _ = p.getBaseVelocity(pybullet_object_id)
        if abs(velocity[0]) > max:
            force_x = 0
        if abs(velocity[1]) > max:
            force_y = 0

        self.move(pybullet_object_id, force_x, force_y)
        
    def get_observation(self, agent_id):
        if agent_id == 'predator':
            pybullet_this_id = self.pybullet_predator_id
            pybullet_other_id = self.pybullet_prey_id
        else:
            pybullet_this_id = self.pybullet_prey_id
            pybullet_other_id = self.pybullet_predator_id

        this_position,_ = p.getBasePositionAndOrientation(pybullet_this_id)
        this_position = np.array(this_position[:2])

        other_position,_ = p.getBasePositionAndOrientation(pybullet_other_id)
        other_position = np.array(other_position[:2])
        return other_position - this_position

    def get_env_state_results(self):
        truncated = False
        rewards = {'predator': 0, 'prey': 0}
        infos = {}   

        predator_pos = self.get_position_predator()
        prey_pos = self.get_position_prey()
        #distance = np.linalg.norm(predator_pos - prey_pos)  # Calculate distance between the two, not used in this example
        if(p.getContactPoints(self.pybullet_predator_id, self.pybullet_prey_id)):
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

        return  rewards, terminated, truncated, infos
    
    def get_position_predator(self):
        predator_pos,_ = p.getBasePositionAndOrientation(self.pybullet_predator_id)
        return np.array([predator_pos[:2]])

    def get_position_prey(self):
        prey_pos,_ = p.getBasePositionAndOrientation(self.pybullet_prey_id)
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

        p.changeDynamics(perimeter_id, -1, lateralFriction=0.5)

        self.create_walled_square(side_length, 0.1, 0.2, height , [0.2, 0.2, 0.2, 1])

    def create_walled_square(self, length, thickness, height, base_height, color):
        """
        Creates a walled square with configurable side length, thickness, height, and color.

        Args:
            length (float): The length of each side of the square.
            thickness (float): The thickness of the walls.
            height (float): The height of the walls.
            color (tuple): A tuple representing the color of the walls in RGB format (r, g, b).
        """
        # Define the dimensions of the wall
        half_length = length / 2
        half_thickness = thickness / 2
        half_height = height / 2

        # Create collision shape for a wall segment
        wall_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[half_length, half_thickness, half_height])

        # Define the visual shape for the wall
        wall_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                        halfExtents=[half_length, half_thickness, half_height],
                                        rgbaColor=[color[0], color[1], color[2], 1])

        # Position and orientation for the four walls
        positions = [
            [0, half_length - half_thickness, base_height + half_height],   # Front wall
            [0, -half_length + half_thickness, base_height + half_height],  # Back wall
            [half_length - half_thickness, 0, base_height + half_height],   # Right wall
            [-half_length + half_thickness, 0, base_height + half_height]   # Left wall
        ]
        orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),                    # Front wall
            p.getQuaternionFromEuler([0, 0, 0]),                    # Back wall
            p.getQuaternionFromEuler([0, 0, 1.5708]),  # Right wall (90 degrees rotation around Z-axis)
            p.getQuaternionFromEuler([0, 0, 1.5708])   # Left wall (90 degrees rotation around Z-axis)
        ]

        # Create walls
        for i in range(4):
            p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=wall_shape,
                            baseVisualShapeIndex=wall_visual,
                            basePosition=positions[i],
                            baseOrientation=orientations[i])
