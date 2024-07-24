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

from pynput import keyboard


class SoccerEnv(BaseMAEnv):

    SIMULATION_STEP_DELAY = 1. / 240.

    def __init__(self, n_team_players=2, max_speed = 0.1, perimeter_side = 10, render=False):
        super(SoccerEnv, self).__init__()
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

        self.pybullet_reds_ids = []
        self.pybullet_blues_ids = []

        for _ in range(n_team_players):
            player_id = p.loadURDF("cube.urdf", [0, 0, 0], useFixedBase=False, globalScaling=0.3)
            self.pybullet_reds_ids.append(player_id)
            p.changeVisualShape(player_id, -1, rgbaColor=[0.8, 0.1, 0.1, 1])

        for _ in range(n_team_players):
            player_id = p.loadURDF("cube.urdf", [0, 0, 0], useFixedBase=False, globalScaling=0.3)
            self.pybullet_blues_ids.append(player_id)
            p.changeVisualShape(player_id, -1, rgbaColor=[0.1, 0.1, 0.8, 1])

        self.pybullet_ball_id = p.loadURDF("sphere2.urdf", [0, 0, 0.5], useFixedBase=False, globalScaling=0.3)
        lateral_friction = 1.0
        spinning_friction = 0.05
        rolling_friction = 0.05
        p.changeDynamics(bodyUniqueId=self.pybullet_ball_id, linkIndex=-1, lateralFriction=lateral_friction, spinningFriction=spinning_friction, rollingFriction=rolling_friction)
        p.changeDynamics(bodyUniqueId=self.pybullet_ball_id, linkIndex=-1, mass=0.1)

        self.perimeter_side = perimeter_side
        self.draw_perimeter(self.perimeter_side)
        vision_length = 20 # Fixed value

        # Create the agents
        # Observation space is the team of the agent (0=red or 1=blue) to know where to score the goal
        # and position of the agent, and velocity x and y
        # and finally the vectors to other agents in the team, the vectors to other agents in the opposite team, and finally the vector to the ball
        n_other_agents = 2*n_team_players - 1

        teams = ['red', 'blue']

        for team in teams:
            for i in range(n_team_players):
                self.register_agent(agent_id=f'{team}_{i}',
                                observation_space=Box(low=np.array([-1, -self.perimeter_side/2,-self.perimeter_side/2, -10, -10] + [-vision_length]*2*(n_other_agents+1)), high=np.array([1, self.perimeter_side/2,self.perimeter_side/2, +10, +10] + [vision_length]*2*(n_other_agents+1)), shape=(5+2*(n_other_agents+1),), dtype=np.float32),
                                action_space=Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float32),
                                model_name='soccer_player'
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
                self.players_touched_ball.add(player_touching_ball)

    def reset(self, seed=0):
        limit_spawn_perimeter = self.perimeter_side / 2 -1
        random_coor = lambda: random.uniform(0, limit_spawn_perimeter)
        for player_id in self.pybullet_reds_ids:
            p.resetBasePositionAndOrientation(player_id, [-random_coor(), 0,  0.5], [0, 0, 0, 1])
            p.resetBaseVelocity(player_id, [0, 0, 0], [0, 0, 0])

        for player_id in self.pybullet_blues_ids:
            p.resetBasePositionAndOrientation(player_id, [random_coor(), 0, 0.5], [0, 0, 0, 1])
            p.resetBaseVelocity(player_id, [0, 0, 0], [0, 0, 0])

        p.resetBasePositionAndOrientation(self.pybullet_ball_id, [0,0, 0.5], [0, 0, 0, 1])
        p.resetBaseVelocity(self.pybullet_ball_id, [0, 0, 0], [0, 0, 0])
        self.sync_wait_for_actions_completion(100)

        obs, _, _, _, info = self.get_state()
        return obs, info

    def step_agent(self, agent_id, action):
        force_x = action[0]
        force_y = action[1]

        pybullet_object_id = self.get_pybullet_id(agent_id)

        # Limit the speed
        max = self.max_speed
        velocity, _ = p.getBaseVelocity(pybullet_object_id)
        velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        if velocity > max:
            #force_x /= (velocity / max)**2
            #force_y /= (velocity / max)**2
            # For impulse-based movements
            force_x = 0
            force_y = 0

            scaling_factor = self.max_speed / velocity
            force_x = scaling_factor * force_x
            force_y = scaling_factor * force_y

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

    def get_observation(self, agent_id):
        # First the team
        if agent_id.startswith('red'):
            obs = [+1]
        else:
            obs = [-1]

        # Now my position
        my_pos = self.get_position(agent_id) # My position
        obs = np.append(obs, my_pos)

        my_pybullet_id = self.get_pybullet_id(agent_id)

        # Now my orientation, nos used finally
        #obs = np.append(obs, self.get_orientation(my_pybullet_id))

        # Now my velocity
        my_velocity, _ = p.getBaseVelocity(my_pybullet_id)
        obs = np.append(obs, my_velocity[:2])

        # Now the other players, first my teammates
        my_team_vectors = []
        other_team_vectors = []
        for other_agent_id in self.agents:
            if other_agent_id != agent_id:
                other_distance_vector = self.get_position(other_agent_id) - my_pos
                same_team = agent_id.split('_')[0] == other_agent_id.split('_')[0]
                if same_team:
                    my_team_vectors.append(other_distance_vector)
                else:
                    other_team_vectors.append(other_distance_vector)
        obs = np.append(obs, my_team_vectors)
        obs = np.append(obs, other_team_vectors)

        # Finally the ball
        ball_pos = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0][:2]
        obs = np.append(obs, ball_pos - my_pos)

        """
        if random.random() < 0.01:
            print(f"Agent {agent_id} observation: {obs}")
            time.sleep(20)
        """

        return obs

    def get_env_state_results(self):
        truncated = False
        rewards = {}
        infos = {}   
        terminated = False

        # Initialize rewards dictionary
        for agent_id in self.agents:
            rewards[agent_id] = 0

        player_out_of_bounds = self.is_player_out_of_bounds()
        goal = self.is_goal()

        if player_out_of_bounds:
            #rewards[player_out_of_bounds] -= 100
            terminated = True
            print(f"Player {player_out_of_bounds} is out of bounds")
        elif goal:
                if goal == 'red':
                    rewards = self.update_reward_team(rewards, 'red', 100)
                    #rewards = self.update_reward_team(rewards, 'blue', -50)
                else:
                    #rewards = self.update_reward_team(rewards, 'red', -50)
                    rewards = self.update_reward_team(rewards, 'blue', 100)
                terminated = True
                print(f"Goal scored by the {goal} team")
        elif self.is_ball_out_of_bounds(): # We must check this after the goal check, becaouse goal is out of bounds
            terminated = True
            print("Ball is out of bounds")
        else:
            # Possesion reward, nor applicable
            """
            player_touching_ball = self.player_touching_ball()
            if player_touching_ball:
                rewards[player_touching_ball] += 0.01 # Possesion reward :-)
            """
            # Ball - Goal proximity reward
            distance_red_goal = self.distance_to_goal('red')
            distance_blue_goal = self.distance_to_goal('blue')
            proximity_reward_red = 0.1 * (0.5 - distance_red_goal / self.perimeter_side)
            proximity_reward_blue = 0.1 * (0.5 - distance_blue_goal / self.perimeter_side)
            rewards = self.update_reward_team(rewards, 'red', proximity_reward_red)
            rewards = self.update_reward_team(rewards, 'blue', proximity_reward_blue)

            """# Player - Ball proximity reward
            for agent_id in self.agents:
                agent_pos = self.get_position(agent_id)
                ball_pos = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0][:2]
                distance = np.linalg.norm(agent_pos - ball_pos)
                # We reward if the distance is less than 25% of the perimeter side
                proximity_reward = 0.05 * (0.25 - distance / self.perimeter_side)
                if proximity_reward > 0:
                    rewards[agent_id] += proximity_reward
            """

            # Ball touched reward
            
            for player_id in self.players_touched_ball:
                #rewards[player_id] += 0.1
                print(f"Player {player_id} touched the ball")
            

        self.show_text(f"Red: {rewards['red_0']:.3f}")
        return rewards, terminated, truncated, infos
    
    def player_touching_ball(self):
        for agent_id in self.agents:
            if p.getContactPoints(self.get_pybullet_id(agent_id), self.pybullet_ball_id):
                return agent_id
        return None
    
    def distance_to_goal(self, goal_line):
        ball_position = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0][:2]
        if goal_line == 'red':
            goal_line_position = [self.perimeter_side / 2, 0]
        else:
            goal_line_position = [-self.perimeter_side / 2, 0]

        return np.linalg.norm(np.array(goal_line_position) - np.array(ball_position))
    
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

        # Check if the ball is completely within the left goal gap
        left_goal_x = -length / 2
        if (ball_x + ball_radius < left_goal_x + thickness and
            -segment_length / 2 < ball_y < segment_length / 2):
            return 'blue'

        # Check if the ball is completely within the right goal gap
        right_goal_x = length / 2
        if (ball_x - ball_radius > right_goal_x - thickness and
            -segment_length / 2 < ball_y < segment_length / 2):
            return 'red'
        
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

    def move(self, pybullet_object_id, force_x, force_y):
        factor = 1000
        force_x *= factor
        force_y *= factor
        #p.applyExternalForce(pybullet_object_id, -1, [force_x, force_y, 0], [0, 0, 0], p.LINK_FRAME)
        pos = p.getBasePositionAndOrientation(pybullet_object_id)[0]
        p.applyExternalForce(pybullet_object_id, -1, [force_x, force_y, 0], pos, p.WORLD_FRAME)

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
        half_size = side_length / 2.0
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

        self.create_pitch(side_length, 0.1, 0.2, height , [0.2, 0.2, 0.2, 1])

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
        gap_size = width * 0.25
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
            p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=long_wall_shape,
                            baseVisualShapeIndex=long_wall_visual,
                            basePosition=positions[i],
                            baseOrientation=orientations[i])
        
        # Short walls with gaps
        for i in range(2, 6):
            p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=short_wall_shape,
                            baseVisualShapeIndex=short_wall_visual,
                            basePosition=positions[i],
                            baseOrientation=orientations[i])


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
