from __future__ import annotations
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np
import random
import os

from ma_sb3 import AgentInfo, BaseMAEnv

# Generate XML for MuJoCo
def generate_mujoco_xml(num_cubes:int=1):

    xml = f"""<mujoco model="swarm_cubes">
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

    # Generate random cubes
    for i in range(num_cubes):
        # In the case of joint slide, the position is relative to the parent body
        # So we need to set the position of the parent body to 0, 0, 0
        x, y = 0, 0
        r, g, b = np.random.rand(3)  # Random color

        xml += f"""
        <body name="cube_{i}" pos="{x} {y} 0.01">
            <geom name="geom_cube_{i}" group="1" type="box" size="0.01 0.01 0.01" rgba="{r} {g} {b} 1" density="5000" friction="0.01 0.01 0.01"/>
            <joint name="cube_{i}_slide_x" type="slide" axis="1 0 0"/> <!-- Move along X -->
            <joint name="cube_{i}_slide_y" type="slide" axis="0 1 0"/> <!-- Move along Y -->
            <joint name="cube_{i}_yaw" type="hinge" axis="0 0 1"/>  <!-- Rotate around Z -->
            <body name="direction_indicator_{i}" pos="0.01 0 0">
                <geom name="indicator_{i}" type="cylinder" size="0.003 0.00001" rgba="1 1 1 1" euler="0 90 0" density="0"/>
            </body>            
        </body>"""

    xml += f"""
            <body name="block" pos="0 0 0.1">
                <joint type="free"/>
                <geom name="geom_block" group="1" type="box" size="0.05 0.05 0.05" rgba="0.9 0.4 0 1" density="{num_cubes*1000}" friction="0.01 0.01 0.01"/>
            </body>
            <body name="target" pos="0 0 0.1">
                <geom name="geom_target" type="cylinder" size="0.15 0.1" rgba="0.0 0.8 0.0 0.4" density="0" contype="0" conaffinity="0" />
            </body>
        </worldbody>
    </mujoco>"""
    
    return xml


class MultiBlockPushRay(MujocoEnv, BaseMAEnv):

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

    def __init__(self, num_cubes = 2, agent_speed=0.5, nrays=5, span_angle_degrees=180, **kwargs):

        default_camera_config = {
            "distance": 2.5,
            "elevation": -10.0,
            "azimuth": 90.0,
            "lookat": [0.0, 0.0, 0.0],
        }

        screen_width = screen_height = 800

        # Overriden initialize_simulation function will use this XML string to create the model instead of model_path 
        self.xml_model = generate_mujoco_xml(num_cubes=num_cubes)

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

        self.nrays = nrays
        self.span_angle_degrees = span_angle_degrees

        for i in range(num_cubes):
            # Observation space
            observation_space = gym.spaces.Box(low=np.array([0, 0, 0]*self.nrays + [-2]*2, dtype=np.float32), high=np.array([1, 1, 1]*self.nrays + [+2]*2, dtype=np.float32), shape=(3*self.nrays+2,))
            # Action space
            action_space = gym.spaces.Box(low=np.array([-1]*2, dtype=np.float32), high=np.array([+1]*2 , dtype=np.float32), shape=(2,))
            self.register_agent(agent_id=f"agcube_{i}",
                            observation_space=observation_space,
                            action_space=action_space,
                            model_name=f"cube"
                            )

        self.agent_speed = agent_speed

        self.block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

        self.mujoco_cube_ids = {}
        self.mujoco_cube_indicator_ids = {}
        self.cubes_components_ids = {}
        self.raycast_rot_mats = {}

        for i in range(num_cubes):
            # Get the ID of the cube
            mujoco_cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{i}")
            # Get the ID of the indicator
            mujoco_cube_indicator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"direction_indicator_{i}")
            # Store the IDs
            self.mujoco_cube_ids[f"agcube_{i}"] = mujoco_cube_id
            self.mujoco_cube_indicator_ids[mujoco_cube_id] = mujoco_cube_indicator_id

            # Get the IDs of the components
            idx_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{i}_slide_x")
            idx_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{i}_slide_y")
            idx_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{i}_yaw")
            self.cubes_components_ids[mujoco_cube_id] = (idx_x, idx_y, idx_yaw)
        
            # Precompute raycast structures
            self.setup_raycast(mujoco_cube_id, self.nrays, self.span_angle_degrees)

        self.rotation_step_size = 0.1
        self.sim_steps_per_decission = 5
        self.active_movements = {}  # Track movements per object

    def do_simulation(self, ctrl, n_frames):
        value = super().do_simulation(ctrl, n_frames)
        if self.render_mode == "human":
            self.render()
        return value

    def get_observation(self, agent_id):
        mujoco_cube_id = self.mujoco_cube_ids[agent_id]
        detected_body_ids, normalized_distances = self.perform_raycast(mujoco_cube_id)
        block_obs = [0, 0, 0] * self.nrays
        for i, detected_body_id in enumerate(detected_body_ids):
            # If the detected body is the block
            if detected_body_id == self.block_id:
                block_obs[i*2] = 1
                block_obs[i*2+2] = normalized_distances[i]
            # If the detected body is a cube different from the agent's cube
            elif detected_body_id in self.mujoco_cube_ids.values() and detected_body_id != mujoco_cube_id:
                block_obs[i*2+1] = 1
                block_obs[i*2+2] = normalized_distances[i]
            # For debugging
            if False and detected_body_id != -1:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, detected_body_id)
                print(f"Ray {i} from {agent_id} hit {body_name} at {normalized_distances[i]}")

        rdv_cube_to_target, _ = self.relative_distance_vector(mujoco_cube_id, self.target_id)

        obs = np.concatenate([block_obs, rdv_cube_to_target[0:2]], dtype=np.float32)
        return obs

    def _sync_wait_for_actions_completion(self):
        all_done = False
        while not all_done:
            all_done = True
            for agent_id in self.agents:
                finished = self.step_move(self.mujoco_cube_ids[agent_id])
                if not finished:
                    all_done = False
            # Execute the simulation step for all elements
            self.do_simulation(self.data.ctrl, self.frame_skip)

    def sync_wait_for_actions_completion(self):
        for _ in range(self.sim_steps_per_decission): #  simulation steps
            for agent_id in self.agents:
                self.step_move(self.mujoco_cube_ids[agent_id])
            # Execute the simulation step for all elements
            self.do_simulation(self.data.ctrl, self.frame_skip)

    def evaluate_env_state(self):
        truncated = False
        rewards = {}
        infos = {}   
        terminated = False

        # Initialize rewards dictionary
        for agent_id in self.agents:
            rewards[agent_id] = 0

        # Check if the block is in the target
        distance_block_target = self.distance_xy(self.block_id, self.target_id )
        if distance_block_target < 0.1 and distance_block_target < self.best_distance:
            print("Target!")
            terminated = True
            for agent_id in self.agents:
                rewards[agent_id] += 100
        else:
            previous_best_distance = self.best_distance
            # Compute individual rewards for each agent
            for agent_id in self.agents:
                mujoco_cube_id = self.mujoco_cube_ids[agent_id]
                distance_agent_block = self.distance_xy(mujoco_cube_id, self.block_id )
                # Individual reward for each agent based on distances except the first state of the episode
                if self.active_movements.get(mujoco_cube_id) is not None:
                    rewards[agent_id] += -self.active_movements[mujoco_cube_id]["distance_done"]
                rewards[agent_id] += -distance_agent_block

                # Update best distance achieved by any agent
                if distance_block_target < self.best_distance:
                    self.best_distance = distance_block_target

            # After individual rewards, add reward based on task (reduced best distance achieved)
            for agent_id in self.agents:
                rewards[agent_id] += (previous_best_distance - self.best_distance)


        return rewards, terminated, truncated, infos

    def reset(self, seed=None):
        super().reset(seed=seed)
        obs, _, _, _, info = self.get_state()
        return obs, info

    def reset_model(self):
        block_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.block_id]]
        self.data.qpos[block_qpos_addr:block_qpos_addr+3] = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.1]

        # As the joint is slide, the qpos is relative to the original location
        for cube_id in range(len(self.mujoco_cube_ids)):
            cube_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[cube_id]]
            self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.1]

        # Necessary to call this function to update the positions before computation
        self.do_simulation(self.data.ctrl, self.frame_skip)

        # Reset the best distance
        for agent_id in self.agents:
            mujoco_id = self.mujoco_cube_ids[agent_id]        
            self.setup_raycast(mujoco_id, self.nrays, self.span_angle_degrees)

        self.best_distance = self.distance_xy(self.block_id, self.target_id)

    def step_agent(self, agent_id:int, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        mujoco_cube_id = self.mujoco_cube_ids[agent_id]   
        self.start_move(mujoco_cube_id, speed=action[0], rotation=action[1])

    def setup_raycast(self, mujoco_cube_id, nrays, angle_covered_degrees):
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
        self.raycast_rot_mats[mujoco_cube_id] = rot_mats

    def perform_raycast(self, mujoco_cube_id, ray_length=10):
        """Performs raycast using stored structures (called at every step)."""

        nrays = self.nrays
        rot_mats = self.raycast_rot_mats[mujoco_cube_id]
        geomgroup = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)  # Detect group 1 (block)
        flg_static = 1 # Include static geoms
        bodyexclude = 0 # Exclude no bodies

        # Get cube position dynamically (it changes over time)
        indicator_id = self.mujoco_cube_indicator_ids[mujoco_cube_id]
        ray_start = np.array(self.data.xpos[indicator_id], dtype=np.float64)  # Updated position

        # Get cube position and orientation
        cube_indicator_mat = self.data.xmat[indicator_id].reshape(3, 3)  # Rotation matrix (3x3)

        # Forward direction (X-axis in local frame)
        world_forward = cube_indicator_mat[:, 0]  # Column 0 is the X direction

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

    def start_move(self, mujoco_cube_id, speed, rotation):
        idx_yaw = self.cubes_components_ids[mujoco_cube_id][2]
        qpos_yaw = self.model.jnt_qposadr[idx_yaw]
        
        yaw_current = self.data.qpos[qpos_yaw]
        yaw_target = yaw_current + rotation
        current_speed = self.agent_speed * speed
        previous_pos = self.data.xpos[mujoco_cube_id].copy()
        
        self.active_movements[mujoco_cube_id] = {
            "yaw_current": yaw_current,
            "yaw_target": yaw_target,
            "remaining_steps": self.sim_steps_per_decission,
            "step_size": rotation/self.sim_steps_per_decission,
            "current_speed": current_speed,
            "distance_done": 0,
            "previous_pos": previous_pos,
            "finished": False
        }

    def step_move(self, mujoco_cube_id):
        # TODO: perform the movement in a fixed number of steps (e.g. 10) and do those simulation steps
        # so that all agents finish at the same time

        if mujoco_cube_id not in self.active_movements:
            self.active_movements[mujoco_cube_id]["finished"] = True
            return True  # No movement for this object
        
        movement = self.active_movements[mujoco_cube_id]
        if movement["remaining_steps"] <= 0:
            self.active_movements[mujoco_cube_id]["finished"] = True
            return True

        idx_x = self.cubes_components_ids[mujoco_cube_id][0]
        idx_y = self.cubes_components_ids[mujoco_cube_id][1]
        idx_yaw = self.cubes_components_ids[mujoco_cube_id][2]
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
        current_pos = self.data.xpos[mujoco_cube_id]
        movement["distance_done"] += np.linalg.norm(current_pos - movement["previous_pos"])
        movement["previous_pos"] = current_pos.copy()

        if movement["remaining_steps"] <= 0:
            self.active_movements[mujoco_cube_id]["finished"] = True
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
