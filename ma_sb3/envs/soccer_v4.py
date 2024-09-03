import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium.spaces import Box
from pynput import keyboard

from ma_sb3 import AgentMAEnv, BaseMAEnv

logger = logging.getLogger(__name__)

class SoccerEnv(BaseMAEnv):

    SIMULATION_STEP_DELAY = 1. / 240.

    # If n_team_players == 0, it is a single player environment
    def __init__(self, n_team_players=2, max_speed = 2, perimeter_side = 10, render=False, record_video_file=None):
        super(SoccerEnv, self).__init__()
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

        self.pybullet_reds_ids = []
        self.pybullet_blues_ids = []

        # Special case for single player environment
        if n_team_players == 0:
            self.single_player = True
            n_team_players = 1
        else:
            self.single_player = False

        for _ in range(n_team_players):
            player_id = p.loadURDF("cube.urdf", [0, 0, 0], useFixedBase=False, globalScaling=0.4)
            self.pybullet_reds_ids.append(player_id)
            p.changeVisualShape(player_id, -1, rgbaColor=[0.8, 0.1, 0.1, 1])
            #p.changeDynamics(bodyUniqueId=player_id, mass=1, linkIndex=-1, lateralFriction=1, spinningFriction=10, rollingFriction=10)
            p.setCollisionFilterGroupMask(player_id, -1, 1, 1)

        if not self.single_player:
            for _ in range(n_team_players):
                player_id = p.loadURDF("cube.urdf", [0, 0, 0], useFixedBase=False, globalScaling=0.4)
                self.pybullet_blues_ids.append(player_id)
                p.changeVisualShape(player_id, -1, rgbaColor=[0.1, 0.1, 0.8, 1])
                #p.changeDynamics(bodyUniqueId=player_id, mass=1, linkIndex=-1, lateralFriction=1, spinningFriction=10, rollingFriction=10)
                p.setCollisionFilterGroupMask(player_id, -1, 1, 1)

        self.pybullet_ball_id = p.loadURDF("soccerball.urdf", [0, 0, 0.5], useFixedBase=False, globalScaling=0.3)

        lateral_friction = 1.0
        spinning_friction = 0.05
        rolling_friction = 0.05
        p.changeDynamics(bodyUniqueId=self.pybullet_ball_id, linkIndex=-1,
                         lateralFriction=lateral_friction, spinningFriction=spinning_friction, rollingFriction=rolling_friction, 
                         mass = 0.1, restitution=0.9) # Restitution is the bounciness of the ball

        self.perimeter_side = perimeter_side
        self.draw_perimeter(self.perimeter_side)
        vision_length = 20 # Fixed value

        # Create the agents
        # Observation space is
        # and rotation and position of the agent, 
        # and the vectors to other agents in the team, the vectors to other agents in the opposite team,
        # and finally the vector to the ball
        # and vector from the ball pos to the goal line
        n_other_agents = 2*n_team_players - 1
        teams = ['red', 'blue']

        # Special case for single player environment
        if self.single_player:            
            n_other_agents = 0
            teams = ['red']
            n_team_players = 1

        for team in teams:
            for i in range(n_team_players):
                self.register_agent(agent_id=f'{team}_{i}',
                                observation_space=Box(low=np.array([-2*math.pi, -self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length]*2*(n_other_agents+2)), high=np.array([+2*math.pi, self.perimeter_side/2,self.perimeter_side/2] + [vision_length]*2*(n_other_agents+2)), shape=(3+2*(n_other_agents+2),), dtype=np.float32),
                                #observation_space=Box(low=np.array([-2*math.pi, -self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length]*2*(n_other_agents+1)), high=np.array([+2*math.pi, self.perimeter_side/2,self.perimeter_side/2] + [vision_length]*2*(n_other_agents+1)), shape=(3+2*(n_other_agents+1),), dtype=np.float32),
                                action_space=Box(low=np.array([-math.pi/18, -1]), high=np.array([+math.pi/18, 1]), shape=(2,), dtype=np.float32),
                                model_name=f"soccer_{team}"
                                )
       
        self.max_speed = max_speed
        self.players_touched_ball = set()

        self.pybullet_text_id = None

        #self.key_control(self.pybullet_reds_ids[0])

    def step_simulation(self):
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def sync_wait_for_actions_completion(self, sim_steps=10):
        self.players_touched_ball = set()
        for _ in range(sim_steps):
            self.step_simulation()
            player_touching_ball = self.player_touching_ball()
            if player_touching_ball is not None:
                self.kick_ball(player_touching_ball)
                self.players_touched_ball.add(player_touching_ball)

    def reset(self, seed=0):

        limit_spawn_perimeter_x = self.perimeter_side / 2 -1
        limit_spawn_perimeter_y = self.perimeter_side / 4 -1

        random_coor_x = lambda: random.uniform(-limit_spawn_perimeter_x, limit_spawn_perimeter_x)
        random_coor_y = lambda: random.uniform(-limit_spawn_perimeter_y, limit_spawn_perimeter_y)

        # Reds score to the right and blues to the left

        # Opposite orientation for the teams
        quat_red = p.getQuaternionFromEuler([0, 0, 0])
        quat_blue = p.getQuaternionFromEuler([0, 0, math.pi])

        for player_id in self.pybullet_reds_ids:
            p.resetBasePositionAndOrientation(player_id, [random_coor_x(), random_coor_y(),  0.5], quat_red)
            p.resetBaseVelocity(player_id, [0, 0, 0], [0, 0, 0])

        for player_id in self.pybullet_blues_ids:
            p.resetBasePositionAndOrientation(player_id, [random_coor_x(), random_coor_y(), 0.5], quat_blue)
            p.resetBaseVelocity(player_id, [0, 0, 0], [0, 0, 0])

        #p.resetBasePositionAndOrientation(self.pybullet_ball_id, [0,0, 0.5], [0, 0, 0, 1])
        # Random ball position version
        p.resetBasePositionAndOrientation(self.pybullet_ball_id, [random_coor_x(), random_coor_y(),  1], [0, 0, 0, 1])
        p.resetBaseVelocity(self.pybullet_ball_id, [0, 0, 0], [0, 0, 0])
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

        # Limit the speed, only applies when inertia is conserved, otherwise velocity is 0
        velocity, _ = p.getBaseVelocity(pybullet_object_id)
        velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        # If velocity is greater than max, cancel the force, only applies when inertia is conserved
        if velocity > self.max_speed:
            force = 0

        self.move(pybullet_object_id, force, self.max_speed)

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

    def get_observation(self, agent_id):
        obs = np.array([])
        my_orientation = self.get_orientation(self.get_pybullet_id(agent_id))
        obs = np.append(obs, my_orientation)

        my_pos = self.get_position(agent_id) # My position
        my_pybullet_id = self.get_pybullet_id(agent_id)
        obs = np.append(obs, my_pos)

        # Now the other players, first my teammates
        my_team_vectors = []
        other_team_vectors = []
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                other_distance_vector = self.get_oriented_distance_vector(my_pybullet_id, self.get_pybullet_id(other_agent_id))[:2]
                same_team = agent_id.split('_')[0] == other_agent_id.split('_')[0]
                if same_team:
                    my_team_vectors.append(other_distance_vector)
                else:
                    other_team_vectors.append(other_distance_vector)
        obs = np.append(obs, my_team_vectors)
        obs = np.append(obs, other_team_vectors)

        # Now the ball
        ball_pos = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0][:2]
        ball_vector = self.get_oriented_distance_vector(my_pybullet_id, self.pybullet_ball_id)[:2]
        obs = np.append(obs, ball_vector)

        """
        if random.random() < 0.01:
            print(f"Agent {agent_id} observation: {obs}")
            time.sleep(20)
        """
        

        # And finally vector from the ball pos to the goal line

        if agent_id.startswith('red'):
            goal_id = self.pybullet_goal_right
        else:
            goal_id = self.pybullet_goal_left

        #goal_line_vector = self.get_oriented_distance_vector(self.pybullet_ball_id, goal_id)[:2]
        goal_line_vector = self.get_oriented_distance_vector(my_pybullet_id, goal_id)[:2]

        obs = np.append(obs, goal_line_vector)

        return obs

    def get_env_state_results(self):
        truncated = False
        rewards = {}
        infos = {}   
        terminated = False

        # Initialize rewards dictionary
        for agent_id in self.agents:
            rewards[agent_id] = 0

        goal = self.is_goal()
        
        """if player_out_of_bounds:
            rewards[player_out_of_bounds] -= 100
            terminated = True
            print(f"Player {player_out_of_bounds} is out of bounds")"""
        if goal:
                if goal == 'red':
                    rewards = self.update_reward_team(rewards, 'red', 100)
                    #rewards = self.update_reward_team(rewards, 'blue', -50)
                else:
                    rewards = self.update_reward_team(rewards, 'blue', 100)
                    #rewards = self.update_reward_team(rewards, 'red', -50)
                if not self.single_player or goal == 'red':
                    terminated = True
                    logger.info(f"Goal scored by the {goal} team")
        elif self.is_ball_out_of_bounds(): # We must check this after the goal check, becaouse goal is out of bounds
            terminated = True
            logger.info("Ball is out of bounds")
        else:
            # Possesion reward
            for agent_id in self.players_touched_ball:
                rewards[agent_id] += 0.1 # Possesion reward :-)
                logger.info(f"Player {agent_id} kicked the ball")
            
        #self.show_text(f"Red: {rewards['red_0']:.3f}")
        return rewards, terminated, truncated, infos
    
    def player_touching_ball(self):
        for agent_id in self.agents:
            if p.getContactPoints(self.get_pybullet_id(agent_id), self.pybullet_ball_id):
                return agent_id
        return None

    def kick_ball(self, agent_id):
        # Get the position of the agent and the ball
        agent_position = self.get_position(agent_id)
        ball_position = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0]

        # Calculate the direction of the kick
        kick_direction = ball_position[:2] - agent_position
        kick_direction = kick_direction / np.linalg.norm(kick_direction)
        kick_direction = np.append(kick_direction, 0)  # Add a zero z-component

        # Apply the kick
        force = 10
        p.applyExternalForce(self.pybullet_ball_id, -1, force * kick_direction, ball_position, p.WORLD_FRAME)

    def is_goal(self):
        """
        Checks if the ball has completely crossed any of the goal gaps.

        Returns:
            None if no goal,
            'blue' if the goal was scored in the left hand side goal gap,
            'red' if scored in the opposite side.
        """
        # Get the position and radius of the ball
        ball_position = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0]
        ball_x = ball_position[0]
        ball_y = ball_position[1]
        ball_radius = 0.2  # Assume this ball_radius
        
        # Define the dimensions based on the pitch creation
        length = self.perimeter_side
        width = self.perimeter_side / 2
        thickness = 0.1
        gap_size = width * 0.25
        segment_length = (width - gap_size) / 2

        # Check if the ball is completely within the right goal gap
        right_goal_x = length / 2
        if (ball_x - ball_radius > right_goal_x - thickness and
            -segment_length / 2 < ball_y < segment_length / 2):
            return 'red'
        
        # Check if the ball is completely within the left goal gap
        left_goal_x = -length / 2
        if (ball_x + ball_radius < left_goal_x + thickness and
            -segment_length / 2 < ball_y < segment_length / 2):
            return 'blue'

        # No goal detected
        return False

    def _is_object_out_of_bounds(self, object_id):  
        margin = 0.5
        pitch_length = self.perimeter_side / 2 + margin
        pitch_width = self.perimeter_side / 4 + margin
        object_pos,_ = p.getBasePositionAndOrientation(object_id)
        object_pos = np.array(object_pos[:2])
        if object_pos[0] < -pitch_length or object_pos[0] > pitch_length or object_pos[1] < -pitch_width or object_pos[1] > pitch_width:
            return True
        return False

    def is_player_out_of_bounds(self):
        for agent_id in self.agents:
            if self._is_object_out_of_bounds(self.get_pybullet_id(agent_id)):
                return agent_id
        return False

    def is_ball_out_of_bounds(self):
        return self._is_object_out_of_bounds(self.pybullet_ball_id)

    def update_reward_team(self, rewards, team, reward):
        for agent_id in self.agents:
            if agent_id.startswith(team):
                rewards[agent_id] += reward
        return rewards

    def get_position(self, agent_id):
        pybullet_id = self.get_pybullet_id(agent_id)
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

    def get_pybullet_id(self, agent_id):
        if agent_id.startswith('red'):
            return int(self.pybullet_reds_ids[int(agent_id.split('_')[-1])])
        else:
            return int(self.pybullet_blues_ids[int(agent_id.split('_')[-1])])

    def render(self, mode='human'):
        pass  # Rendering handled in real-time if GUI mode is enabled

    def close(self):
        p.disconnect()

    def draw_perimeter(self, side_length):
        height = 0.4
        half_size = side_length / 2.0 + 0.6
        half_height = height / 2.0
        color = [0, 0.4, 0, 0.9]  # Green color

        # Create visual shape for the box
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size/2, half_height],
            rgbaColor=color,
            specularColor=[0, 0, 0]
        )

        # Create collision shape for the box
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size/2, half_height]
        )

        # Create the multi-body object
        perimeter_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, half_height]
        )

        p.changeDynamics(perimeter_id, -1, lateralFriction=0.5)

        #self.create_pitch(side_length, 0.1, 0.2, height , [0.2, 0.2, 0.2, 1])
        self.create_pitch(side_length, 0.1, 0.4, height , [0.5, 0.5, 0.5, 1])

    def create_pitch(self, length, thickness, height, base_height, color):
        """
        Creates a rectangular structure with walls, with gaps in the center of the shorter sides.

        Args:
            length (float): The length of the rectangle (double the width).
            width (float): The width of the rectangle (half the length).
            thickness (float): The thickness of the walls.
            height (float): The height of the walls.
            base_height (float): The base height at which the walls are positioned.
            color (tuple): A tuple representing the color of the walls in RGB format (r, g, b).
        """
        # Define half dimensions for easier calculations
        width = length / 2
        half_length = length / 2
        half_width = width / 2
        half_thickness = thickness / 2
        half_height = height / 2
        gap_size = width * 0.40
        gap_size = 2
        segment_length = (width - gap_size) / 2

        # Create collision shape for wall segments
        long_wall_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                halfExtents=[half_length, half_thickness, half_height])
        short_wall_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                halfExtents=[segment_length / 2, half_thickness, half_height])

        # Define the visual shape for the walls
        long_wall_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[half_length, half_thickness, half_height],
                                            rgbaColor=[color[0], color[1], color[2], 1])
        short_wall_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                halfExtents=[segment_length / 2, half_thickness, half_height],
                                                rgbaColor=[color[0], color[1], color[2], 1])

        # Position and orientation for the walls
        positions = [
            [0, -half_width + half_thickness, base_height + half_height],   # Front wall
            [0, +half_width - half_thickness, base_height + half_height],  # Back wall
            [half_length - half_thickness, half_width - segment_length / 2 - half_thickness, base_height + half_height],  # Right wall (segment 1)
            [half_length - half_thickness, -half_width + segment_length / 2 + half_thickness, base_height + half_height],  # Right wall (segment 2)
            [-half_length + half_thickness, half_width - segment_length / 2 - half_thickness, base_height + half_height],  # Left wall (segment 1)
            [-half_length + half_thickness, -half_width + segment_length / 2 + half_thickness, base_height + half_height]  # Left wall (segment 2)
        ]
        orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),                    # Front wall
            p.getQuaternionFromEuler([0, 0, 0]),                    # Back wall
            p.getQuaternionFromEuler([0, 0, 1.5708]),  # Right wall (90 degrees rotation around Z-axis)
            p.getQuaternionFromEuler([0, 0, 1.5708]),  # Right wall (90 degrees rotation around Z-axis)
            p.getQuaternionFromEuler([0, 0, 1.5708]),  # Left wall (90 degrees rotation around Z-axis)
            p.getQuaternionFromEuler([0, 0, 1.5708])   # Left wall (90 degrees rotation around Z-axis)
        ]

        # Create walls
        # Long walls
        for i in range(2):
            id = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=long_wall_shape,
                            baseVisualShapeIndex=long_wall_visual,
                            basePosition=positions[i],
                            baseOrientation=orientations[i])
            p.setCollisionFilterGroupMask(id, -1, 2, 1)
            p.changeDynamics(id, -1, restitution=0.8) # Bounciness


        # Short walls with gaps
        for i in range(2, 6):
            id = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=short_wall_shape,
                            baseVisualShapeIndex=short_wall_visual,
                            basePosition=positions[i],
                            baseOrientation=orientations[i])
            p.setCollisionFilterGroupMask(id, -1, 2, 1)
            p.changeDynamics(id, -1, restitution=0.8) # Bounciness

        self.pybullet_goal_right = self.create_goal([half_length - 0.15, 0, base_height], p.getQuaternionFromEuler([0, 0, 0]))
        self.pybullet_goal_left = self.create_goal([-half_length + 0.15, 0, base_height], p.getQuaternionFromEuler([0, 0, math.pi]))

    def create_goal(self, position, orientation):
        script_dir = os.path.dirname(__file__)
        goal_path = os.path.join(script_dir, "./meshes/goal.obj")

        scaling_factor = [1, 1, 1]
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=goal_path, meshScale=scaling_factor)

        # Add a collision shape for physics simulation (optional)
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=goal_path, meshScale=scaling_factor, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)

        # Create a multibody object that combines both visual and collision shapes
        goal_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=orientation)
        #goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=orientation)

        p.changeDynamics(goal_id, -1, restitution=0.8) # Bounciness

        return goal_id




    def show_text(self, text):
        if self.pybullet_text_id is not None:
            p.removeUserDebugItem(self.pybullet_text_id)
        self.pybullet_text_id = p.addUserDebugText(text, [0, -4, 2], textColorRGB=[0, 0, 0], textSize=2)


    def key_control(self, object_id):
        # Define the key press handler
        def on_press(key):
            try:
                # Convert key to character
                k = key.char
            except AttributeError:
                k = key.name
            
            # Define the force to be applied
            force_magnitude = 100
            force = [0, 0, 0]
            
            # Check which key is pressed and set the force accordingly
            if key == keyboard.Key.up:  # Apply force forward
                force = [0, force_magnitude, 0]
            elif key == keyboard.Key.down:  # Apply force backward
                force = [0, -force_magnitude, 0]
            elif key == keyboard.Key.left:  # Apply force to the left
                force = [-force_magnitude, 0, 0]
            elif key == keyboard.Key.right:  # Apply force to the right
                force = [force_magnitude, 0, 0]
            else:
                return
            
            # Apply the force to the object
            p.applyExternalForce(objectUniqueId=object_id, 
                                linkIndex=-1,  # -1 applies the force to the base
                                forceObj=force, 
                                posObj=p.getBasePositionAndOrientation(object_id)[0],  # Apply the force at the object's position
                                flags=p.WORLD_FRAME)
        
        # Create a keyboard listener
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        # Keep the script running to listen for key presses
        try:
            while True:
                p.stepSimulation()
                # Get the current position of the object
                pos, _ = p.getBasePositionAndOrientation(object_id)
                
                # Print the position with 2 decimal places
                formatted_pos = [f"{coord:.2f}" for coord in pos]
                print("Position:", formatted_pos)
        except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
