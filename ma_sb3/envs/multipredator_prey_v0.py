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

class MultiPredatorPreyMAEnv(BaseMAEnv):

    SIMULATION_STEP_DELAY = 1. / 240.

    def __init__(self, n_predators=2, max_speed_predator = 5, max_speed_prey = 5, perimeter_side = 10, render=False):
        super(MultiPredatorPreyMAEnv, self).__init__()
        self.render_mode = render
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
        self.draw_perimeter(self.perimeter_side)
        vision_length = 20 # Fixed value

        # Create the agents
        obs_space_others_len = 2 * (n_predators - 1) + 2 # x,y per other predator + x,y of prey
        for i in range(n_predators):
            self.register_agent(agent_id=f'predator_{i}',
                            observation_space=Box(low=np.array([-self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length]*2*(n_predators-1+1)), high=np.array([self.perimeter_side/2,self.perimeter_side/2] + [vision_length]*2*(n_predators-1+1)), shape=(2+2*(n_predators-1+1),), dtype=np.float32),
                            #observation_space=Box(low=np.array([-self.perimeter_side/2]*2*(n_predators+1)), high=np.array([self.perimeter_side/2]*2*(n_predators+1)), shape=(2*(n_predators+1),), dtype=np.float32),
                            #observation_space=Box(low=np.array([-self.perimeter_side/2]*2*(2+1)), high=np.array([self.perimeter_side/2]*2*(2+1)), shape=(2*(2+1),), dtype=np.float32),
                            action_space=Box(low=np.array([-10, -10]), high=np.array([10, 10]), shape=(2,), dtype=np.float32),
                            model_name='predator'
                            )

        obs_space_others_len = 2 * n_predators # x,y per each predator
        self.register_agent(agent_id='prey',
                        observation_space=Box(low=np.array([-self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length]*2*n_predators), high=np.array([self.perimeter_side/2,self.perimeter_side/2] + [vision_length]*2*n_predators), shape=(2+2*n_predators,), dtype=np.float32),
                        #observation_space=Box(low=np.array([-self.perimeter_side/2]*2*(n_predators+1)), high=np.array([self.perimeter_side/2]*2*(n_predators+1)), shape=(2*(n_predators+1),), dtype=np.float32),
                        action_space=Box(low=np.array([-10, -10]), high=np.array([10, 10]), shape=(2,), dtype=np.float32),
                        model_name='prey'
                        )
        
        self.max_speed_predator = max_speed_predator
        self.max_speed_prey = max_speed_prey

    def step_simulation(self):
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def sync_wait_for_actions_completion(self, sim_steps=10):
        for _ in range(sim_steps):
            self.step_simulation()
             # We must check collisions at each simulation step, and stop if the prey is captured
            if self.predator_touching_prey():
                break
        
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
        force_x = action[0]
        force_y = action[1]

        if agent_id.startswith('predator'):
            pybullet_object_id = self.get_pybullet_id(agent_id)
            max = self.max_speed_predator
        else:
            pybullet_object_id = self.pybullet_prey_id
            max = self.max_speed_prey

        # Limit the speed
        velocity, _ = p.getBaseVelocity(pybullet_object_id)
        velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        if velocity > max:
            force_x /= (velocity / max)**2
            force_y /= (velocity / max)**2
            # For impulse-based movements
            #force_x = 0
            #force_y = 0

        self.move(pybullet_object_id, force_x, force_y)

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

    def _limited_get_observation(self, agent_id):
        my_pos = self.get_position(agent_id) # My position
        pybullet_agent_id = self.get_pybullet_id(agent_id)
        obs = my_pos
        if agent_id.startswith('predator'):
            other_predators_by_distance = self.get_object_distances(pybullet_agent_id, self.pybullet_predators_ids)
            closest_predator_pybullet_id = other_predators_by_distance[0][0]
            closest_orientation_vector = self.get_oriented_distance_vector(pybullet_agent_id, closest_predator_pybullet_id)[:2]
            obs = np.append(obs, closest_orientation_vector)
            obs_prey = self.get_oriented_distance_vector(pybullet_agent_id, self.pybullet_prey_id)[:2]
            obs = np.append(obs, obs_prey)
        else: # if prey
            for other_agent_id in self.agents:
                if other_agent_id != agent_id:
                    other_orientation_vector = self.get_oriented_distance_vector(pybullet_agent_id, self.get_pybullet_id(other_agent_id))[:2]
                    obs = np.append(obs, other_orientation_vector)
        return obs

    def get_observation(self, agent_id):
        my_pos = self.get_position(agent_id) # My position
        my_pybullet_id = self.get_pybullet_id(agent_id)
        obs = my_pos
        # self.agents is a dictionary, but according to the spec the order of the agents should be preserved
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                #other_pos = self.get_position(other_agent_id)
                other_distance_vector = self.get_oriented_distance_vector(my_pybullet_id, self.get_pybullet_id(other_agent_id))[:2]
                obs = np.append(obs, other_distance_vector)
        return obs
        """
        my_pos = self.get_position(agent_id)
        obs = my_pos
        # self.agents is a dictionary, but according to the spec the order of the agents should be preserved
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                distance_vector_to_other = self.get_oriented_distance_vector(self.get_pybullet_id(agent_id), self.get_pybullet_id(other_agent_id))[:2]
                obs = np.append(obs, distance_vector_to_other)
        return obs
        """

    def evaluate_env_state(self):
        truncated = False
        rewards = {}
        infos = {}   

        predator_touching_prey = self.predator_touching_prey()

        if predator_touching_prey is not None:
            # Predators win
            terminated = True
            rewards.update(self.reward_predators(+5))
            #rewards.update(self.reward_predators(+0)) # No reward for the predators
            rewards[predator_touching_prey] += 10 # Predator that touched the prey gets extra reward
            rewards['prey'] = -10
            print(f"Captured by {predator_touching_prey}")
        else:        
            predator_out = self.is_any_predator_out_of_bounds()
            limit_perimeter = self.perimeter_side / 2
            prey_pos = self.get_position('prey')
            prey_out = np.any(np.logical_or(prey_pos < -limit_perimeter, prey_pos > limit_perimeter))
            if predator_out:
                terminated = True
                rewards.update(self.reward_predators(-100))
                print("Predator: Out of bounds!")
                rewards['prey'] = 10 # Prey wins
            elif prey_out:
                terminated = True
                rewards['prey'] = -100
                print("Prey: Out of bounds!")
                rewards.update(self.reward_predators(+10)) # Predator wins
            else:
                # prey wins +1 per time step, predator loses 1?
                #rewards.update(self.reward_predators(-0.1))
                rewards.update(self.reward_predators(0)) # Set to 0
                rewards['prey'] = 0.1
                terminated = False

        return rewards, terminated, truncated, infos
    
    def predator_touching_prey(self):
        for agent_id in self.agents:
            if agent_id.startswith('predator'):
                if p.getContactPoints(self.get_pybullet_id(agent_id), self.pybullet_prey_id):
                    return agent_id
        return None
    
    def is_any_predator_out_of_bounds(self):
        limit_perimeter = self.perimeter_side / 2
        for predator_id in self.pybullet_predators_ids:
            predator_pos,_ = p.getBasePositionAndOrientation(predator_id)
            predator_pos = np.array([predator_pos[:2]])
            if np.any(np.logical_or(predator_pos < -limit_perimeter, predator_pos > limit_perimeter)):
                return True
        return False

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

    def move(self, pybullet_object_id, force_x, force_y):
        factor = 100
        force_x *= factor
        force_y *= factor
        p.applyExternalForce(pybullet_object_id, -1, [force_x, force_y, 0], [0, 0, 0], p.LINK_FRAME)

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

