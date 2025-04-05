from ma_sb3.envs.base_swarm_env import BaseSwarmEnv
import numpy as np
import mujoco
import random
import math

class SwarmProtectAssetEnv(BaseSwarmEnv):
    def __init__(self, num_assets = 1, circularity_required = 0.99, *args, **kwargs):
        self.num_assets = num_assets # Important at this point as it is used to generate the XML in the parent class
        
        super().__init__(*args, **kwargs)

        self.asset_ids = []
        for i in range(num_assets):
            self.asset_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"asset_{i}"))

        self.circularity_required = circularity_required
        self.movement = False

    def reset_model(self):
        # Create the asset in a random position
        radius = 0.5
        random_pos = lambda: (
            (lambda theta, d: (d * math.cos(theta), d * math.sin(theta)))
            (random.uniform(0, 2 * math.pi), radius * math.sqrt(random.uniform(0, 1)))
        )

        # Generate random positions for all assets
        for asset_id in self.asset_ids:
            x, y = random_pos()

            asset_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[asset_id]]
            self.data.qpos[asset_qpos_addr:asset_qpos_addr+3] = [x, y, 0.1]

        # Now generate the agents
        super().reset_model()



    def get_env_state_results(self):
        truncated = False
        rewards = {}
        infos = {}   
        terminated = False

        # Initialize rewards dictionary
        for agent_id in self.agents:
            rewards[agent_id] = 0

        # Step 1: Assign each agent to the closest asset
        asset_groups = {i: {} for i in range(len(self.asset_ids))}
        asset_positions = [
            self.data.xpos[asset_id][:2] for asset_id in self.asset_ids
        ]

        agent_closest_asset = {}  # Map agent_id -> closest asset index

        for agent_id in self.agents:
            agent_pos = self.data.xpos[self.mujoco_cube_ids[agent_id]][:2]
            distances = [np.linalg.norm(agent_pos - asset_pos) for asset_pos in asset_positions]
            closest_asset_idx = int(np.argmin(distances))
            asset_groups[closest_asset_idx][agent_id] = agent_pos
            agent_closest_asset[agent_id] = closest_asset_idx

        # Step 2: Evaluate asset protection conditions
        asset_distance_required = 0.2
        distance_score_required = 0.95

        assets_protection_achieved = []
        assets_circularity_scores = []
        assets_distance_scores = []

        for i, asset_id in enumerate(self.asset_ids):
            asset_x, asset_y = self.data.xpos[asset_id][:2]
            agent_positions = asset_groups[i]

            agent_positions_array = np.array(list(agent_positions.values()))

            if len(agent_positions) <= 1:
                this_circularity = 0
                this_distance_scores = [0] * len(agent_positions_array)
            else:
                this_circularity, this_distance_scores, this_angular_scores = self.circle_fit_score(
                    agent_positions_array, asset_x, asset_y, asset_distance_required
                )

                
            assets_circularity_scores.append(this_circularity)
            assets_distance_scores.append(this_distance_scores)

            this_avg_distance_score = np.mean(this_distance_scores) if len(this_distance_scores) > 0 else 0

            this_protection_achieved = (
                this_circularity >= self.circularity_required and
                this_avg_distance_score > distance_score_required
            )
            assets_protection_achieved.append(this_protection_achieved)

            if this_protection_achieved:
                print(f"Achieved protection for asset {i}: {this_circularity}!")

        # Step 3: Assign rewards to each agent
        for agent_id in self.agents:
            mujoco_body_id = self.mujoco_cube_ids[agent_id]

            if self.active_movements.get(mujoco_body_id) is not None:
                rewards[agent_id] -= self.active_movements[mujoco_body_id]['distance_done']
                rewards[agent_id] -= self.active_movements[mujoco_body_id]['rotation_done'] / 10

            agent_asset_group_index = agent_closest_asset[agent_id]
            agent_index_in_group = list(asset_groups[agent_asset_group_index].keys()).index(agent_id)

            # Use only the distance score of this agent from the correct group
            agent_distance_score = assets_distance_scores[agent_asset_group_index][agent_index_in_group]
            rewards[agent_id] += agent_distance_score / 10

            if agent_distance_score > distance_score_required:
                rewards[agent_id] += 0.5 * assets_circularity_scores[agent_asset_group_index]

            # If the agent is protecting an asset
            if assets_protection_achieved[agent_asset_group_index]:
                rewards[agent_id] += 0.5

            # If all assets are protected, give a bonus
            if all(assets_protection_achieved):
                rewards[agent_id] += 1.0

        if all(assets_protection_achieved):
            print("All assets are protected!")

        if self.movement:
            self.push_assets_random()

        return rewards, terminated, truncated, infos


    def get_observation(self, agent_id):
        mujoco_cube_id = self.mujoco_cube_ids[agent_id]
        detected_body_ids, normalized_distances = self.perform_raycast(mujoco_cube_id)
        ray_obs = np.zeros((self.nrays, 3+self.communication_items), dtype=np.float32)
        for i, detected_body_id in enumerate(detected_body_ids):
            if detected_body_id in self.asset_ids:
                ray_obs[i][0] = 1
                ray_obs[i][2] = normalized_distances[i]
            # If the detected body is a cube different from the agent's cube
            if detected_body_id in self.mujoco_cube_ids.values() and detected_body_id != mujoco_cube_id:
                ray_obs[i][1] = 1 # Flag for cube detected
                ray_obs[i][2] = normalized_distances[i]
                if self.communication_items > 0:
                    ray_obs[i][3:3+self.communication_items] = self.agent_comm_messages[detected_body_id] # Communication message from the agent
            # For debugging
            if False and detected_body_id != -1 and detected_body_id != mujoco_cube_id:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, detected_body_id)
                print(f"Ray {i} from {agent_id} hit {body_name} at {normalized_distances[i]}")

        ray_obs = ray_obs.flatten()
        obs = ray_obs.copy()
        return obs

    def circle_fit_score(self, points, xc, yc, r):
        N = len(points)
                
        # Convert points to numpy array
        points = np.array(points)
        
        # Compute distances from center
        distances = np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)
        
        # Radial deviation score (1 means all points are on the circle)
        radial_deviation = np.abs(distances - r)
        radial_score = np.exp(-np.var(distances - r))
        
        # Compute angles
        angles = np.arctan2(points[:, 1] - yc, points[:, 0] - xc)
        sorted_indices = np.argsort(angles)
        angles_sorted = angles[sorted_indices]
        
        # Compute pairwise angle differences
        angle_diffs = np.diff(np.concatenate(([angles_sorted[-1] - 2 * np.pi], angles_sorted)))
        
        # Expected uniform angle spacing
        expected_spacing = 2 * np.pi / N
        
        # Angular uniformity score
        angular_score = np.exp(-np.var(angle_diffs - expected_spacing))
        
        # Compute individual point scores
        distance_scores = np.exp(-radial_deviation)  # Higher means closer to ideal radius
        
        # Normalize angular distances
        angular_distances = np.abs(angle_diffs - expected_spacing)
        angular_scores = np.exp(-angular_distances)  # Higher means closer to uniform spacing
        
        # Final score as geometric mean
        final_score = np.sqrt(radial_score * angular_score)

        # Reorder with original indices
        angular_scores[sorted_indices] = angular_scores
        
        return final_score, distance_scores, angular_scores


    def set_camera_at(self, x, y):
        self.mujoco_renderer.viewer.cam.lookat[0] = x
        self.mujoco_renderer.viewer.cam.lookat[1] = y

    def push_assets_random(self):
        for asset_id in self.asset_ids:
            self.data.xfrc_applied[asset_id, :] = 0  # Reset external forces
            # Apply force (only first three elements, last three are torque)
            force = np.random.uniform(-0.5, 0.5, size=3)  # Random force in x, y
            force[2] = 0
            self.data.xfrc_applied[asset_id, :3] = force  

    # Generate XML for MuJoCo
    def generate_mujoco_xml(self, num_cubes:int=1):

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
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="10 10" reflectance="0.05"/>
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


        for i in range(self.num_assets):
            x, y = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            xml += f"""       
                <body name="asset_{i}" pos="{x} {y} 0.1">
                    <joint type="free"/> 
                    <geom name="geom_asset_{i}" group="1" type="cylinder" size="0.05 0.05" rgba="0.0 0.8 0.0 0.4" density="5000" friction="0.01 0.01 0.01"/>
                </body>"""
        
        xml += f"""
            </worldbody>
        </mujoco>"""
        
        return xml
