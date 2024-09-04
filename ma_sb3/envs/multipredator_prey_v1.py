from concurrent.futures import ThreadPoolExecutor
from ma_sb3 import AgentMAEnv, BaseMAEnv

from gymnasium.spaces import Box
import numpy as np

import pybullet as p
import pybullet_data
import time

import pybullet as p
import random
import os
import math

import logging

logger = logging.getLogger(__name__)
class MultiPredatorPreyMAEnv(BaseMAEnv):

    SIMULATION_STEP_DELAY = 1. / 240.

    def __init__(self, n_predators=2, max_speed_predator = 2, max_speed_prey = 4, reward_all_predators = 10, reward_catching_predator = 5, perimeter_side = 10, render=False, record_video_file=None):
        super(MultiPredatorPreyMAEnv, self).__init__()
        self.render_mode = render
        # For video recording
        if record_video_file is not None:
            self.physicsClient = p.connect(p.GUI if render else p.DIRECT, options=f"--mp4='{record_video_file}'")
        else:
            self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        # Hide debug panels in PyBullet
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        # Reorient the debug camera
        p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0,-1.5,0])

        # Load the plane and objects
        self.pybullet_plane_id = p.loadURDF("plane.urdf", [0,0,0])

        self.pybullet_predators_ids = []
        for _ in range(n_predators):
            predator_id = p.loadURDF("cube.urdf", [0, 0, 0], useFixedBase=False, globalScaling=0.3)
            self.pybullet_predators_ids.append(predator_id)
            p.changeVisualShape(predator_id, -1, rgbaColor=[0.8, 0.1, 0.1, 1])

        self.pybullet_prey_id = p.loadURDF("cube.urdf", [1, 1, 0], useFixedBase=False, globalScaling=0.2)
        #p.changeDynamics(self.pybullet_predator_id, -1, lateralFriction=1)
        #p.changeDynamics(self.pybullet_prey_id, -1, lateralFriction=1)
        self.perimeter_side = perimeter_side
        self.draw_floor(self.perimeter_side)
        vision_length = 20 # Fixed value

        # Create the agents
        for i in range(n_predators):
            self.register_agent(agent_id=f'predator_{i}',
                            observation_space=Box(low=np.array([-2*math.pi, -self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length]*2*(n_predators-1+1)), high=np.array([+2*math.pi, self.perimeter_side/2,self.perimeter_side/2] + [vision_length]*2*(n_predators-1+1)), shape=(3+2*(n_predators-1+1),), dtype=np.float32),
                            action_space=Box(low=np.array([-math.pi/18, -1]), high=np.array([+math.pi/18, 1]), shape=(2,), dtype=np.float32),
                            model_name='predator'
                            )

        self.register_agent(agent_id='prey',
                        observation_space=Box(low=np.array([-2*math.pi, -self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length]*2*n_predators), high=np.array([+2*math.pi, self.perimeter_side/2,self.perimeter_side/2] + [vision_length]*2*n_predators), shape=(3+2*n_predators,), dtype=np.float32),
                        action_space=Box(low=np.array([-math.pi/18, -1]), high=np.array([+math.pi/18, 1]), shape=(2,), dtype=np.float32),
                        model_name='prey'
                        )
        
        self.max_speed_predator = max_speed_predator
        self.max_speed_prey = max_speed_prey
        self.reward_all_predators = reward_all_predators
        self.reward_catching_predator = reward_catching_predator

    def step_simulation(self):
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def sync_wait_for_actions_completion(self, sim_steps=10):
        self.predators_touching_prey = set()
        for _ in range(sim_steps):
            self.step_simulation()
            predator_touching_prey = self.get_predators_touching_prey()
            if predator_touching_prey is not None:
                self.predators_touching_prey.add(predator_touching_prey)

        # This is the visual optimal point for raycast removal
        if self.render_mode:
            p.removeAllUserDebugItems()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        limit_spawn_perimeter = self.perimeter_side / 2 -1
        random_coor = lambda: random.uniform(-limit_spawn_perimeter, limit_spawn_perimeter)
        for predator_id in self.pybullet_predators_ids:
            p.resetBasePositionAndOrientation(predator_id, [random_coor(), random_coor(), 0.5], [0, 0, 0, 1])
            p.resetBaseVelocity(predator_id, [0, 0, 0], [0, 0, 0])
        p.resetBasePositionAndOrientation(self.pybullet_prey_id, [random_coor(), random_coor(), 0.5], [0, 0, 0, 1])
        p.resetBaseVelocity(self.pybullet_prey_id, [0, 0, 0], [0, 0, 0])
        self.sync_wait_for_actions_completion(100)

        obs, _, _, _, info = self.get_state()
        return obs, info

    def step_agent(self, agent_id, action):

        rotation_offset = action[0]
        force = action[1]

        pybullet_object_id = self.get_pybullet_id(agent_id)

        # Rotate the player towards the force direction
        position, _ = p.getBasePositionAndOrientation(pybullet_object_id)

        #linear_velocity, angular_velocity = p.getBaseVelocity(pybullet_object_id) # Conserving inertia for the next step
        linear_velocity = angular_velocity = [0,0,0] # Starting still in next step

        angle = self.get_orientation(pybullet_object_id)
        angle += rotation_offset
        p.resetBasePositionAndOrientation(pybullet_object_id, position, p.getQuaternionFromEuler([0, 0, angle]))
        p.resetBaseVelocity(pybullet_object_id, linearVelocity=linear_velocity, angularVelocity=angular_velocity)

        max_speed = self.max_speed_predator if agent_id.startswith('predator') else self.max_speed_prey

        # Limit the speed, only applies when inertia is conserved, otherwise velocity is 0
        velocity, _ = p.getBaseVelocity(pybullet_object_id)
        velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        # If velocity is greater than max, cancel the force, only applies when inertia is conserved
        if velocity > max_speed:
            force = 0

        self.move(pybullet_object_id, force, max_speed)

    def move(self, pybullet_object_id, force, nax_speed):
        factor = 1000
        force *= factor * nax_speed
        force = [force, 0, 0]
        position, orientation = p.getBasePositionAndOrientation(pybullet_object_id)
        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # Transform the force vector from local to world coordinates
        force_world = np.dot(rotation_matrix, force)

        p.applyExternalForce(pybullet_object_id, -1, force_world, position, p.WORLD_FRAME)

    def get_oriented_distance_vector(self, object_1_id, object_2_id):
        # Get the base position and orientation of object_1
        base_position_1, base_orientation_1 = p.getBasePositionAndOrientation(object_1_id)

        # Get the base position of object_2
        position_2, _ = p.getBasePositionAndOrientation(object_2_id)

        # Convert the position of object_2 to the base frame of object_1
        # Compute the inverse of the base's world orientation
        base_rotation_matrix_1 = np.array(p.getMatrixFromQuaternion(base_orientation_1)).reshape(3, 3)

        # Compute the relative position of object_2 in the world frame
        relative_position = np.array(position_2) - np.array(base_position_1)

        # Transform the relative position to the base frame of object_1
        relative_position_base_frame = np.dot(base_rotation_matrix_1.T, relative_position)
        
        return relative_position_base_frame

    def get_orientation(self, pybullet_object_id):
        # Get the orientation of the object
        _, orientation = p.getBasePositionAndOrientation(pybullet_object_id)
        
        # Convert quaternion to euler angles
        euler_angles = p.getEulerFromQuaternion(orientation)
        
        # Extract the angle around the z-axis
        # euler_angles returns (roll, pitch, yaw)
        # yaw is the angle around the z-axis
        angle_z = euler_angles[2]  # radians
        
        return angle_z
        
    def get_observation(self, agent_id):
        obs = np.array([])
        my_orientation = self.get_orientation(self.get_pybullet_id(agent_id))
        obs = np.append(obs, my_orientation)

        my_pos = self.get_position(agent_id) # My position
        my_pybullet_id = self.get_pybullet_id(agent_id)
        obs = np.append(obs, my_pos)
        
        # self.agents is a dictionary, but according to the spec the order of the agents should be preserved
        # so first we get the other predators, and finally the prey
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                other_distance_vector = self.get_oriented_distance_vector(my_pybullet_id, self.get_pybullet_id(other_agent_id))[:2]
                obs = np.append(obs, other_distance_vector)
        return obs

    def get_env_state_results(self):
        truncated = False
        rewards = {}
        infos = {}

        predator_touching_prey = self.predators_touching_prey.pop() if len(self.predators_touching_prey) > 0 else None

        if predator_touching_prey is not None:
            # Predators win
            terminated = True
            rewards.update(self.reward_predators(self.reward_all_predators))
            rewards[predator_touching_prey] += self.reward_catching_predator # Predator that touched the prey gets extra reward
            rewards['prey'] = -10
            logger.info(f"Captured by {predator_touching_prey}")
            infos['winner'] = "predator"
        else:        
            predator_out = self.predator_out_of_bounds()
            limit_perimeter = self.perimeter_side / 2
            prey_pos = self.get_position('prey')
            prey_out = np.any(np.logical_or(prey_pos < -limit_perimeter, prey_pos > limit_perimeter))
            if predator_out is not None:
                terminated = True
                rewards.update(self.reward_predators(0)) # Initialized to 0
                rewards[predator_out] = -100 # This predator is penalized
                logger.info("Predator: Out of bounds!")
                rewards['prey'] = +10 # Prey wins
            elif prey_out:
                terminated = True
                rewards['prey'] = -100
                logger.info("Prey: Out of bounds!")
                rewards.update(self.reward_predators(self.reward_all_predators)) # Predators wins
            else:
                # prey wins some per time step
                rewards.update(self.reward_predators(0)) # Set to 0
                rewards['prey'] = 0.1
                terminated = False

        return rewards, terminated, truncated, infos
    
    def get_predators_touching_prey(self):
        for agent_id in self.agents:
            if agent_id.startswith('predator'):
                if p.getContactPoints(self.get_pybullet_id(agent_id), self.pybullet_prey_id):
                    return agent_id
        return None
    

    def predator_out_of_bounds(self):
        limit_perimeter = self.perimeter_side / 2
        for agent_id in self.agents:
            if agent_id.startswith('predator'):
                predator_pos = self.get_position(agent_id)
                if np.any(np.logical_or(predator_pos < -limit_perimeter, predator_pos > limit_perimeter)):
                    return agent_id
        return None

    def reward_predators(self, reward):
        rewards = {}
        for agent_id in self.agents:
            if agent_id.startswith('predator'):
                rewards[agent_id] = reward
        return rewards

    def get_position(self, agent_id):
        if agent_id.startswith('predator'):
            pybullet_id = self.get_pybullet_id(agent_id)
        else:
            pybullet_id = self.pybullet_prey_id

        pos,_ = p.getBasePositionAndOrientation(pybullet_id)
        return np.array(pos[:2])

    def get_object_distances(self, this_object_id, object_ids):
        distances = []
        pos1, _ = p.getBasePositionAndOrientation(this_object_id)
        for obj_id in object_ids:
            if obj_id != this_object_id:  # Avoid comparing the object to itself
                pos2, _ = p.getBasePositionAndOrientation(obj_id)
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                distances.append((obj_id, pos2, distance))
        
        # Sort distances based on the second element (distance)
        distances.sort(key=lambda x: x[2])
        
        return distances
    
    def get_pybullet_id(self, agent_id):
        if agent_id.startswith('predator'):
            return int(self.pybullet_predators_ids[int(agent_id.split('_')[-1])])
        else:
            return self.pybullet_prey_id

    def render(self, mode='human'):
        pass  # Rendering handled in real-time if GUI mode is enabled

    def close(self):
        p.disconnect()

    def draw_floor(self, side_length):
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

        self.create_walled_perimeter(side_length, 0.1, 0.2, height , [0.2, 0.2, 0.2, 1])

    def create_walled_perimeter(self, length, thickness, height, base_height, color):
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

