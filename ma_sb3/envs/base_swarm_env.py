from __future__ import annotations
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np
import random
import os
import math

from ma_sb3 import AgentInfo, BaseMAEnv

class BaseSwarmEnv(MujocoEnv, BaseMAEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple",],
    }
    
    # Overriden initialize_simulation function will use this XML string to create the model instead of model_path
    def _initialize_simulation(self,):
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`.
        """
        model = mujoco.MjModel.from_xml_string(self.xml_model) # This line is the difference
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def __init__(self, num_robots=2, agent_speed=0.5, forward_only = True, nrays=5, span_angle_degrees=180,
                 individual_comm_items=0, env_comm_items=0, comm_learning=True,
                 obs_body_prefixes = [],
                 **kwargs):

        default_camera_config = {
            "distance": 2.5,
            "elevation": -10.0,
            "azimuth": 90.0,
            "lookat": [0.0, 0.0, 0.0],
        }

        screen_width = screen_height = 800

        # Overriden initialize_simulation function will use this XML string to create the model instead of model_path 
        self.xml_model = self.generate_mujoco_xml(num_robots=num_robots)

        MujocoEnv.__init__(
            self,
            model_path=os.path.abspath(__file__), # Dummy value, not used, but it must be a valid path
            frame_skip=5,
            observation_space=None,
            default_camera_config=default_camera_config,
            width=screen_width,
            height=screen_height,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        BaseMAEnv.__init__(self)

        self.nrays = nrays
        self.span_angle_degrees = span_angle_degrees
        self.individual_comm_items = individual_comm_items
        self.comm_learning = comm_learning
        self.env_comm_items = env_comm_items
        self.forward_only = forward_only
        self.obs_body_prefixes = obs_body_prefixes
        obs_num_bodies = len(obs_body_prefixes)
        
        for i in range(num_robots):
            # Observation space
            # Per every raycast: distance to body, then one-hot for each type of body and individual communication items
            # Then general communication items
            observation_space = gym.spaces.Box(
                low=np.array(([0] + [0]*obs_num_bodies + [-1] * self.individual_comm_items) * self.nrays + [-1] * self.env_comm_items, dtype=np.float32),
                high=np.array(([1] + [1]*obs_num_bodies + [1] * self.individual_comm_items) * self.nrays + [1] * self.env_comm_items, dtype=np.float32),
                shape=((1 + obs_num_bodies + self.individual_comm_items)*self.nrays + self.env_comm_items,))
            # Action space
            low_limit = 0 if self.forward_only else -1
            # Check if in comm is in the action space and should be learned
            if self.comm_learning:
                self.action_comm_items = self.individual_comm_items
            else:
                self.action_comm_items = 0
            action_space = gym.spaces.Box(
                low=np.array([low_limit]*2 + [-1] * self.action_comm_items, dtype=np.float32),
                high=np.array([+1]*(2+self.action_comm_items) , dtype=np.float32),
                shape=(2+self.action_comm_items,))
            
            self.register_agent(agent_id=f"agrobot_{i}",
                            observation_space=observation_space,
                            action_space=action_space,
                            model_name=f"robot"
                            )

        self.agent_speed = agent_speed

        self.mujoco_robot_ids = {}
        self.mujoco_robot_indicator_ids = {}
        self.robots_components_ids = {}
        self.raycast_rot_mats = {}

        for i in range(num_robots):
            # Get the ID of the robot
            mujoco_robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{i}")
            # Get the ID of the indicator
            mujoco_robot_indicator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"direction_indicator_{i}")
            # Store the IDs
            self.mujoco_robot_ids[f"agrobot_{i}"] = mujoco_robot_id
            self.mujoco_robot_indicator_ids[mujoco_robot_id] = mujoco_robot_indicator_id

            # Get the IDs of the components
            idx_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{i}_slide_x")
            idx_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{i}_slide_y")
            idx_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{i}_yaw")
            self.robots_components_ids[mujoco_robot_id] = (idx_x, idx_y, idx_yaw)
        
            # Precompute raycast structures
            self.setup_raycast(mujoco_robot_id, self.nrays, self.span_angle_degrees)

        self.rotation_step_size = 0.1
        self.sim_steps_per_decission = 5
        self.active_movements = {}  # Track movements per object

        # Initialize the agents public communication message
        self.agent_comm_messages = {mujoco_robot_id: [0]*self.individual_comm_items for mujoco_robot_id in self.mujoco_robot_ids.values()}

    def do_simulation(self, ctrl, n_frames):
        value = super().do_simulation(ctrl, n_frames)
        if self.render_mode == "human":
            self.render()
        return value

    def get_observation(self, agent_id):
        self_mujoco_robot_id = self.mujoco_robot_ids[agent_id]
        detected_body_ids, normalized_distances = self.perform_raycast(self_mujoco_robot_id)
        obs_num_bodies = len(self.obs_body_prefixes)
        ray_obs = np.zeros((self.nrays, obs_num_bodies + 1 + self.individual_comm_items), dtype=np.float32)
        for i, detected_body_id in enumerate(detected_body_ids):
            # If the detected body is a robot different from the agent's robot
            if detected_body_id != self_mujoco_robot_id:
                # Check if the object prefix is in the list
                for j, prefix in enumerate(self.obs_body_prefixes):
                    if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, detected_body_id).startswith(prefix):
                        ray_obs[i][j] = 1
                ray_obs[i][obs_num_bodies] = normalized_distances[i]
                if self.individual_comm_items > 0:
                    ray_obs[i][obs_num_bodies+1:obs_num_bodies+1+self.individual_comm_items] = self.agent_comm_messages[detected_body_id] # Communication message from the agent
            # For debugging
            if False and detected_body_id != -1 and detected_body_id != self_mujoco_robot_id:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, detected_body_id)
                print(f"Ray {i} from {agent_id} hit {body_name} at {normalized_distances[i]}")

        ray_obs = ray_obs.flatten()
        obs = ray_obs.copy()
        return obs

    def sync_wait_for_actions_completion(self):
        for _ in range(self.sim_steps_per_decission): #  simulation steps
            for agent_id in self.agents:
                self.step_move(self.mujoco_robot_ids[agent_id])
            # Execute the simulation step for all elements
            self.do_simulation(self.data.ctrl, self.frame_skip)

    def evaluate_env_state(self):
        raise NotImplementedError

    def dissapear_body(self, body_id):
        body_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
        self.data.qpos[body_qpos_addr:body_qpos_addr+3] = [0,0,-1000]

    def reset(self, seed=None):
        super().reset(seed=seed)
        obs, _, _, _, info = self.get_state()
        return obs, info

    def reset_model(self):
        radius = 0.5
        random_pos = lambda: (
            (lambda theta, d: (d * math.cos(theta), d * math.sin(theta)))
            (random.uniform(0, 2 * math.pi), radius * math.sqrt(random.uniform(0, 1)))
        )

        # Spawn the robots in random positions
        # As the joint is slide, the qpos is relative to the original location
        for robot_id in self.mujoco_robot_ids.values():
            robot_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[robot_id]]
            self.data.qpos[robot_qpos_addr:robot_qpos_addr+3] = [*random_pos(), 0.1]

        # Necessary to call this function to update the positions before computation
        self.do_simulation(self.data.ctrl, self.frame_skip)

        # Reset the best distance
        for agent_id in self.agents:
            mujoco_id = self.mujoco_robot_ids[agent_id]        
            self.setup_raycast(mujoco_id, self.nrays, self.span_angle_degrees)

        # Initialize the agents public communication message
        self.agent_comm_messages = {mujoco_robot_id: [0]*self.individual_comm_items for mujoco_robot_id in self.mujoco_robot_ids.values()}

    def step_agent(self, agent_id:int, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        mujoco_robot_id = self.mujoco_robot_ids[agent_id]
        self.start_move(mujoco_robot_id, speed=action[0], rotation=action[1])
        if self.comm_learning:
            self.agent_comm_messages[mujoco_robot_id] = action[2:2+self.action_comm_items] # To be displayed and used by other agents

    def setup_raycast(self, mujoco_robot_id, nrays, angle_covered_degrees):
        """Precomputes raycast structures (only called once)."""

        # Generate evenly spaced ray directions
        angles = np.linspace(-angle_covered_degrees / 2, angle_covered_degrees / 2, nrays)  # Spread symmetrically
        if nrays == 1:
            angles = [0]  # Single ray at angle 0

        rot_mats = np.zeros((nrays, 2, 2), dtype=np.float64)
        for i, angle in enumerate(angles):
            # Compute rotated direction using 2D rotation matrix (Z-axis rotation)
            rot_mats[i] = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])

        # Store raycast structure
        self.raycast_rot_mats[mujoco_robot_id] = rot_mats

    def perform_raycast(self, mujoco_robot_id, ray_length=10):
        """Performs raycast using stored structures (called at every step)."""

        nrays = self.nrays
        rot_mats = self.raycast_rot_mats[mujoco_robot_id]
        geomgroup = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)  # Detect group 1
        flg_static = 1 # Include static geoms
        bodyexclude = 0 # Exclude no bodies

        # Get robot position dynamically (it changes over time)
        indicator_id = self.mujoco_robot_indicator_ids[mujoco_robot_id]
        ray_start = np.array(self.data.xpos[indicator_id], dtype=np.float64)  # Updated position

        # Get robot position and orientation
        robot_indicator_mat = self.data.xmat[indicator_id].reshape(3, 3)  # Rotation matrix (3x3)

        # Forward direction (X-axis in local frame)
        world_forward = robot_indicator_mat[:, 0]  # Column 0 is the X direction

        # Apply offset to the ray start position based on the forward orientation
        forward_offset = 1e-6 * world_forward
        ray_start += forward_offset

        ray_dirs = np.zeros((nrays, 3), dtype=np.float64)

        for i, rot_mat in enumerate(rot_mats):
            rotated_dir = rot_mat @ world_forward[:2]  # Apply rotation to X, Y components
            ray_dirs[i] = np.array([rotated_dir[0], rotated_dir[1], world_forward[2]]) * ray_length  # Scale rays

        # Output arrays
        geomids = np.full(nrays, -1, dtype=np.int32)  # No hit by default
        fractions = np.zeros(nrays, dtype=np.float64)

        mujoco.mj_multiRay(
            self.model,
            self.data,
            ray_start.flatten(),
            ray_dirs.flatten(),  
            geomgroup.flatten(),  
            flg_static,
            bodyexclude,
            geomids, 
            fractions,
            nrays,
            ray_length 
        )

        return geomids, fractions  # Arrays of hit geom IDs and fractions

    def start_move(self, mujoco_robot_id, speed, rotation):
        idx_yaw = self.robots_components_ids[mujoco_robot_id][2]
        qpos_yaw = self.model.jnt_qposadr[idx_yaw]
        
        yaw_current = self.data.qpos[qpos_yaw]
        yaw_target = yaw_current + rotation
        current_speed = self.agent_speed * speed
        previous_pos = self.data.xpos[mujoco_robot_id].copy()
        
        self.active_movements[mujoco_robot_id] = {
            "yaw_original": yaw_current,
            "yaw_current": yaw_current,
            "yaw_target": yaw_target,
            "remaining_steps": self.sim_steps_per_decission,
            "step_size": rotation/self.sim_steps_per_decission,
            "current_speed": current_speed,
            "distance_done": 0,
            "rotation_done": 0,
            "previous_pos": previous_pos,
            "finished": False
        }

    def step_move(self, mujoco_robot_id):
        # TODO: perform the movement in a fixed number of steps (e.g. 10) and do those simulation steps
        # so that all agents finish at the same time

        if mujoco_robot_id not in self.active_movements:
            self.active_movements[mujoco_robot_id]["finished"] = True
            return True  # No movement for this object
        
        movement = self.active_movements[mujoco_robot_id]
        if movement["remaining_steps"] <= 0:
            self.active_movements[mujoco_robot_id]["finished"] = True
            return True

        idx_x = self.robots_components_ids[mujoco_robot_id][0]
        idx_y = self.robots_components_ids[mujoco_robot_id][1]
        idx_yaw = self.robots_components_ids[mujoco_robot_id][2]
        qpos_yaw = self.model.jnt_qposadr[idx_yaw]

        # Determine step size
        #step_size = min(self.rotation_step_size, movement["remaining_rotation"])
        step_size = movement["step_size"]
        movement["remaining_steps"] -= 1

        movement["yaw_current"] += step_size
        self.data.qpos[qpos_yaw] = movement["yaw_current"]

        # Compute new velocity
        vx_new = movement["current_speed"] * np.cos(movement["yaw_current"])
        vy_new = movement["current_speed"] * np.sin(movement["yaw_current"])

        # Apply new velocities
        self.data.qvel[idx_x] = vx_new
        self.data.qvel[idx_y] = vy_new

        # Update distance done
        current_pos = self.data.xpos[mujoco_robot_id]
        movement["distance_done"] += np.linalg.norm(current_pos - movement["previous_pos"])
        movement["previous_pos"] = current_pos.copy()
        movement["rotation_done"] = abs(movement["yaw_current"] - movement["yaw_original"])

        if movement["remaining_steps"] <= 0:
            self.active_movements[mujoco_robot_id]["finished"] = True
            return True
        else:
            return False

    def distance_xy(self, body_id_1, body_id_2):
        """
        Compute the Euclidean distance between two objects in the X-Y plane.
        
        Args:
            body_id_1: The ID of the first body.
            body_id_2: The ID of the second body.
        
        Returns:
            The Euclidean distance between the two objects in the X-Y plane.
        """

        # Get positions
        pos1 = self.data.xpos[body_id_1]
        pos2 = self.data.xpos[body_id_2]

        distance = np.linalg.norm(pos1[0:2] - pos2[0:2])
        return distance

    def relative_distance_vector(self, body_id_1, body_id_2):
        """
        Compute the distance vector from object 1 to object 2 in object 1's local frame,
        and also return the difference in yaw (orientation) between the two bodies.
        
        Args:
            body_id_1: The ID of the first body (reference).
            body_id_2: The ID of the second body.
        
        Returns:
            A tuple: 
                - A NumPy array representing the distance vector in object 1's local frame.
                - A float representing the difference in yaw (orientation) between the two bodies.
        """
        if body_id_1 == -1 or body_id_2 == -1:
            raise ValueError("Invalid body IDs. Ensure both objects exist in the model.")

        # Get positions
        pos1 = self.data.xpos[body_id_1]  # (x, y, z) position of Object 1
        pos2 = self.data.xpos[body_id_2]  # (x, y, z) position of Object 2

        # Compute global distance vector
        distance_vector = pos2 - pos1  # (dx, dy, dz)

        # Get quaternion of Object 1
        quat1 = self.data.xquat[body_id_1]
        # Compute yaw angle for body 1
        yaw1 = np.arctan2(2 * (quat1[0] * quat1[3] + quat1[1] * quat1[2]),  
                        1 - 2 * (quat1[2]**2 + quat1[3]**2))

        # Get quaternion of Object 2
        quat2 = self.data.xquat[body_id_2]
        # Compute yaw angle for body 2
        yaw2 = np.arctan2(2 * (quat2[0] * quat2[3] + quat2[1] * quat2[2]),  
                        1 - 2 * (quat2[2]**2 + quat2[3]**2))

        # Compute the rotation matrix for body 1's yaw
        rot_matrix1 = np.array([
            [np.cos(-yaw1), -np.sin(-yaw1), 0],  # Rotate in the opposite direction
            [np.sin(-yaw1),  np.cos(-yaw1), 0],
            [0,             0,            1]  # Z-axis remains unchanged
        ])

        # Transform distance vector into Object 1's local frame
        local_distance_vector = rot_matrix1 @ distance_vector

        # Compute the yaw difference between body 1 and body 2
        yaw_difference = yaw2 - yaw1
        # Normalize the yaw difference to be between -pi and pi
        yaw_difference = (yaw_difference + np.pi) % (2 * np.pi) - np.pi

        return local_distance_vector, yaw_difference

    # Generate XML for MuJoCo
    def generate_mujoco_xml(self, num_robots:int=1):

        xml = f"""<mujoco model="swarm_robots">
        <option timestep="0.01" gravity="0 0 -9.81"/>
        <visual>
            <headlight diffuse="0.8 0.8 0.8" ambient="0.8 0.8 0.8" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
        </visual>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"
            markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.05"/>
        </asset>
        <worldbody>
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="0.01 0.01 0.01"/>"""

        # Generate random robots
        for i in range(num_robots):
            # In the case of joint slide, the position is relative to the parent body
            # So we need to set the position of the parent body to 0, 0, 0
            x, y = 0, 0
            r, g, b = np.random.rand(3)  # Random color

            xml += f"""
            <body name="robot_{i}" pos="{x} {y} 0.01">
                <geom name="geom_robot_{i}" group="1" type="box" size="0.01 0.01 0.01" rgba="{r} {g} {b} 1" density="5000" friction="0.01 0.01 0.01"/>
                <joint name="robot_{i}_slide_x" type="slide" axis="1 0 0"/> <!-- Move along X -->
                <joint name="robot_{i}_slide_y" type="slide" axis="0 1 0"/> <!-- Move along Y -->
                <joint name="robot_{i}_yaw" type="hinge" axis="0 0 1"/>  <!-- Rotate around Z -->
                <body name="direction_indicator_{i}" pos="0.01 0 0">
                    <geom name="indicator_{i}" type="cylinder" size="0.003 0.00001" rgba="1 1 1 1" euler="0 90 0" density="0"/>
                </body>            
            </body>"""

        xml += f"""        
            </worldbody>
        </mujoco>"""
        
        return xml
