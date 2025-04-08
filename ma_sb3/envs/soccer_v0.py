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

from ma_sb3 import AgentMAEnv, BaseMAEnv
from ma_sb3.envs.soccer_stadium import create_stadium, create_player, create_ball

logger = logging.getLogger(__name__)

class SoccerEnv(BaseMAEnv):

    SIMULATION_STEP_DELAY = 1. / 240.

    # If n_team_players == 0, it is a single player environment
    def __init__(self, n_team_players=2, single_team=False, max_speed = 1, perimeter_side = 10, render=None, record_video_file=None):
        super(SoccerEnv, self).__init__()
        self.render_mode = render
        # For video recording
        if record_video_file is not None:
            self.physicsClient = p.connect(p.GUI if render=='human' else p.DIRECT, options=f"--mp4='{record_video_file}'")
        else:
            self.physicsClient = p.connect(p.GUI if render=='human' else p.DIRECT)

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

        # Create the stadium
        self.perimeter_side = perimeter_side
        pybullet_goal_ids = create_stadium(self.perimeter_side)
        self.pybullet_goal_right_id = pybullet_goal_ids[0]
        self.pybullet_goal_left_id = pybullet_goal_ids[1]

        # Create the players
        self.pybullet_reds_ids = []
        self.pybullet_blues_ids = []

        red_color = [0.8, 0.1, 0.1, 1]
        blue_color = [0.1, 0.1, 0.8, 1]

        self.single_team = single_team

        for _ in range(n_team_players):
            player_id = create_player([0, 0, 0], red_color)
            self.pybullet_reds_ids.append(player_id)

        if not self.single_team:
            for _ in range(n_team_players):
                player_id = create_player([0, 0, 0], blue_color)
                self.pybullet_blues_ids.append(player_id)

        self.pybullet_ball_id = create_ball([0, 0, 0])

        vision_length = self.perimeter_side

        # Create the agents
        # Observation space is
        # and rotation and position of the agent, 
        # and the vectors to other agents in the team, the vectors to other agents in the opposite team,
        # and finally the vector to the ball
        # and vector from the ball pos to the goal line

        if not self.single_team:
            n_other_agents = 2*n_team_players - 1
            teams = ['red', 'blue']
        else:
            n_other_agents = n_team_players - 1
            teams = ['red']

        for team in teams:
            for i in range(n_team_players):
                self.register_agent(agent_id=f'{team}_{i}',
                                #observation_space=Box(low=np.array([-2*math.pi, -self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length]*2*(n_other_agents+2)), high=np.array([+2*math.pi, self.perimeter_side/2,self.perimeter_side/2] + [vision_length]*2*(n_other_agents+2)), shape=(3+2*(n_other_agents+2),), dtype=np.float32),
                                observation_space=Box(low=np.array([-2*math.pi,-10, -self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length]*2*(n_other_agents+1)), high=np.array([+2*math.pi,+10, self.perimeter_side/2,self.perimeter_side/2] + [vision_length]*2*(n_other_agents+1)), shape=(4+2*(n_other_agents+1),), dtype=np.float32),
                                #observation_space=Box(low=np.array([-2*math.pi] + [-self.perimeter_side/2,-self.perimeter_side/2]*2), high=np.array([+2*math.pi] + [self.perimeter_side/2,self.perimeter_side/2]*2), shape=(5,), dtype=np.float32),
                                action_space=Box(low=np.array([-1, -1]), high=np.array([+1, +1]), shape=(2,), dtype=np.float32),
                                model_name=f"soccer_{team}"
                                )
       
        self.max_speed = max_speed
        self.players_touched_ball = set()

        self.pybullet_text_id = None


    def step_simulation(self):
        p.stepSimulation()
        if self.render_mode:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def sync_wait_for_actions_completion(self, max_sim_steps=500):
        self.players_touched_ball = set()
        for step in range(max_sim_steps):
            self.step_simulation()
            player_touching_ball = self.player_touching_ball()
            if player_touching_ball is not None:
                self.kick_ball(player_touching_ball)
                self.players_touched_ball.add(player_touching_ball)
            # Every 10 sim steps, check status
            if step % 10 == 0:
                if self.is_goal() or self.is_ball_out_of_bounds():
                    return
                # If the players are not moving, return
                players_velocity = [np.linalg.norm(p.getBaseVelocity(player_id)[0]) for player_id in self.pybullet_reds_ids + self.pybullet_blues_ids]
                if all([v < 2*self.max_speed for v in players_velocity]):
                    return


    def reset(self, seed=None):
        super().reset(seed=seed)

        limit_spawn_perimeter_x = self.perimeter_side / 2 -1
        limit_spawn_perimeter_y = self.perimeter_side / 4 -1

        #random_coor_x = lambda: random.uniform(-limit_spawn_perimeter_x, limit_spawn_perimeter_x)
        random_coor_x = lambda: random.uniform(0, limit_spawn_perimeter_x)
        random_coor_y = lambda: random.uniform(-limit_spawn_perimeter_y, limit_spawn_perimeter_y)

        # Reds score to the right and blues to the left

        # Opposite orientation for the teams
        quat_red = p.getQuaternionFromEuler([0, 0, math.pi])
        quat_blue = p.getQuaternionFromEuler([0, 0, 0])

        for player_id in self.pybullet_reds_ids:
            p.resetBasePositionAndOrientation(player_id, [-random_coor_x(), random_coor_y(),  0.5], quat_red)
            p.resetBaseVelocity(player_id, [0, 0, 0], [0, 0, 0])

        for player_id in self.pybullet_blues_ids:
            p.resetBasePositionAndOrientation(player_id, [+random_coor_x(), random_coor_y(), 0.5], quat_blue)
            p.resetBaseVelocity(player_id, [0, 0, 0], [0, 0, 0])

        p.resetBasePositionAndOrientation(self.pybullet_ball_id, [0,0, 0.5], [0, 0, 0, 1])
        # Random ball position version
        #p.resetBasePositionAndOrientation(self.pybullet_ball_id, [random_coor_x(), random_coor_y(),  1], [0, 0, 0, 1])
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

        rotation_offset = rotation_offset * math.pi / 6 # 30 degrees max rotation
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

    def move(self, pybullet_object_id, force, max_speed):
        factor = 1000
        force *= factor * max_speed
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
        pybullet_agent_id = self.get_pybullet_id(agent_id)
        obs = np.array([])
        my_orientation = self.get_orientation(pybullet_agent_id)
        obs = np.append(obs, my_orientation)

        # Add speed
        velocity, _ = p.getBaseVelocity(pybullet_agent_id)
        velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        obs = np.concatenate((obs, [velocity]), dtype=np.float32)

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

        # Order the vectors by distance (to facilitate learning)
        #my_team_vectors.sort(key=lambda x: np.linalg.norm(x))
        #other_team_vectors.sort(key=lambda x: np.linalg.norm(x))
        # Add the vectors to the observation
        obs = np.append(obs, my_team_vectors)
        obs = np.append(obs, other_team_vectors)

        # Now the ball
        ball_pos = p.getBasePositionAndOrientation(self.pybullet_ball_id)[0][:2]
        ball_vector = self.get_oriented_distance_vector(my_pybullet_id, self.pybullet_ball_id)[:2]
        obs = np.append(obs, ball_vector)
        #obs = np.append(obs, ball_pos)

        """
        if random.random() < 0.01:
            print(f"Agent {agent_id} observation: {obs}")
            time.sleep(20)
        """
        

        # And finally vector from the agent to the goal line

        if agent_id.startswith('red'):
            goal_id = self.pybullet_goal_right_id
        else:
            goal_id = self.pybullet_goal_left_id

        goal_line_vector = self.get_oriented_distance_vector(my_pybullet_id, goal_id)[:2]

        #obs = np.append(obs, goal_line_vector)

        return obs

    def evaluate_env_state(self):
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
                if not self.single_team or goal == 'red':
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
            
        #Time penalty
        rewards = self.update_reward_team(rewards, 'red', -0.1)
        rewards = self.update_reward_team(rewards, 'blue', -0.1)
        #self.show_text(f"Red: {rewards['red_0']:.3f}")
        return rewards, terminated, truncated, infos
    
    def player_touching_ball(self):
        for agent_id in self.agents:
            if p.getContactPoints(self.get_pybullet_id(agent_id), self.pybullet_ball_id):
                return agent_id
        return None

    def kick_ball(self, agent_id):
        return
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

    def show_text(self, text):
        if self.pybullet_text_id is not None:
            p.removeUserDebugItem(self.pybullet_text_id)
        self.pybullet_text_id = p.addUserDebugText(text, [0, -4, 2], textColorRGB=[0, 0, 0], textSize=2)

