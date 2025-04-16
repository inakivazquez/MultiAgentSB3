from ma_sb3.envs.base_swarm_env import BaseSwarmEnv
import numpy as np
import mujoco
import random
import math

class SwarmProtectAssetEnv(BaseSwarmEnv):
    def __init__(self, num_assets = 1, surrounding_required = 0.99, *args, **kwargs):
        self.num_assets = num_assets # Important at this point as it is used to generate the XML in the parent class
        
        super().__init__(*args, **kwargs)

        self.asset_ids = []
        for i in range(num_assets):
            self.asset_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"asset_{i}"))

        self.surrounding_required = surrounding_required
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


    def evaluate_env_state(self):
        truncated = False
        rewards = {}
        infos = {}   
        terminated = False

        agent_ids = list(self.agents)

        # Initialize rewards dictionary
        rewards = dict.fromkeys(agent_ids, 0.0)

        # Step 1: Calculate asset possitions and agent positions
        asset_positions = np.array([self.data.xpos[aid][:2] for aid in self.asset_ids])
        agent_positions = np.array([
            self.data.xpos[self.mujoco_cube_ids[aid]][:2] for aid in agent_ids
        ])

        # Step 1.1: Compute closest assets to agents
        assets_close_to_agents_indices, assets_close_to_agents_distances = self.compute_closest_assets_to_agents(agent_positions, asset_positions)

        # Step 2: Evaluate asset protection conditions
        asset_distance_required = 0.2
        asset_distance_margin = 0.1
        distance_score_required = 0.9

        assets_protection_achieved = []
        assets_surrounding_scores = []

        for i, asset_pos in enumerate(asset_positions):
            score = self.compute_surrounding_score(
                asset_pos,
                agent_positions,
                asset_distance_required - asset_distance_margin,
                asset_distance_required + asset_distance_margin
            )
            assets_surrounding_scores.append(score)
            protection = score >= self.surrounding_required
            assets_protection_achieved.append(protection)

            if protection:
                print(f"Achieved protection for asset {i}: {score}!")

        all_protected = all(assets_protection_achieved)

        # Step 3: Compute agent rewards
        for i, agent_id in enumerate(agent_ids):
            body_id = self.mujoco_cube_ids[agent_id]

            move = self.active_movements.get(body_id)
            if move:
                rewards[agent_id] -= move['distance_done']
                rewards[agent_id] -= move['rotation_done'] / 10

            closest_idx = assets_close_to_agents_indices[i]
            closest_dist = assets_close_to_agents_distances[i]
            dist_score = np.exp(-abs(closest_dist - asset_distance_required))
            
            # Reward the agent for being close to the required distance to the asset
            rewards[agent_id] += dist_score / 10

            # If the agent has better distance score than required, bonus based on surrouding score
            if dist_score >= distance_score_required:
                rewards[agent_id] += 0.5 * assets_surrounding_scores[closest_idx]

            # If the agent is participating in the successful protection of the asset, bonus
            if assets_protection_achieved[closest_idx]:
                rewards[agent_id] += 0.5

            # If all assets are protected, give a bonus
            if all_protected:
                rewards[agent_id] += 1.0

        if all_protected:
            print("All assets are protected!")

        if self.movement:
            self.push_assets_random()

        return rewards, terminated, truncated, infos


    def get_observation(self, agent_id):
        mujoco_cube_id = self.mujoco_cube_ids[agent_id]
        detected_body_ids, normalized_distances = self.perform_raycast(mujoco_cube_id)
        ray_obs = np.zeros((self.nrays, 3+self.individual_comm_items), dtype=np.float32)
        for i, detected_body_id in enumerate(detected_body_ids):
            if detected_body_id in self.asset_ids:
                ray_obs[i][0] = 1
                ray_obs[i][2] = normalized_distances[i]
            # If the detected body is a cube different from the agent's cube
            if detected_body_id in self.mujoco_cube_ids.values() and detected_body_id != mujoco_cube_id:
                ray_obs[i][1] = 1 # Flag for cube detected
                ray_obs[i][2] = normalized_distances[i]
                if self.individual_comm_items > 0:
                    ray_obs[i][3:3+self.individual_comm_items] = self.agent_comm_messages[detected_body_id] # Communication message from the agent
            # For debugging
            if False and detected_body_id != -1 and detected_body_id != mujoco_cube_id:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, detected_body_id)
                print(f"Ray {i} from {agent_id} hit {body_name} at {normalized_distances[i]}")

        ray_obs = ray_obs.flatten()
        obs = ray_obs.copy()
        return obs

    def compute_closest_assets_to_agents(self, agent_positions, asset_positions):
        # agent_positions: (N_agents, 2)
        # asset_positions: (N_assets, 2)
        agent_positions = np.asarray(agent_positions)
        asset_positions = np.asarray(asset_positions)
        
        # Expand dimensions to compute pairwise distances:
        # agents -> shape (N_agents, 1, 2)
        # assets -> shape (1, N_assets, 2)
        diffs = agent_positions[:, np.newaxis, :] - asset_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)  # shape (N_agents, N_assets)

        # Get closest asset index and distance per agent
        closest_indices = np.argmin(distances, axis=1)
        closest_distances = np.min(distances, axis=1)

        return closest_indices.astype(int), closest_distances.astype(float)

    def compute_surrounding_score(self, asset_pos, agents_pos, min_distance, max_distance):
        agents_pos = np.asarray(agents_pos)
        dx = agents_pos[:, 0] - asset_pos[0]
        dy = agents_pos[:, 1] - asset_pos[1]
        distances = np.hypot(dx, dy)
        mask = (distances >= min_distance) & (distances <= max_distance)

        if np.sum(mask) < 2:
            return 0.0

        angles = np.arctan2(dy[mask], dx[mask])
        angles = np.sort(angles)
        angles = np.append(angles, angles[0] + 2 * np.pi)

        max_gap = np.max(np.diff(angles))
        return 1 - (max_gap / (2 * np.pi))


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
    def generate_mujoco_xml(self, num_cubes:int=1, shape:str='disc'):

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

        <asset>
            <mesh name="dome_mesh" file="/home/inaki/masb3/ma_sb3/envs/meshes/dome/dome.obj"  scale="1 1 0.5"/>
            <mesh name="disc_dome_mesh" file="/home/inaki/masb3/ma_sb3/envs/meshes/dome/disc_with_dome.obj"  scale="1 1 0.5"/>
        </asset>

        <worldbody>
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="0.01 0.01 0.01"/>"""

        # Generate random cubes
        for i in range(num_cubes):
            # In the case of joint slide, the position is relative to the parent body
            # So we need to set the position of the parent body to 0, 0, 0
            x, y = 0, 0
            r, g, b = np.random.rand(3)  # Random color

            if shape == 'cube':
                xml += f"""
                <body name="cube_{i}" pos="{x} {y} 0.01">
                    <geom name="geom_cube_{i}" group="1" type="box" size="0.01 0.01 0.01"
                        rgba="{r} {g} {b} 1" density="5000" friction="0.01 0.01 0.01"/>
                    <joint name="cube_{i}_slide_x" type="slide" axis="1 0 0"/> <!-- Move along X -->
                    <joint name="cube_{i}_slide_y" type="slide" axis="0 1 0"/> <!-- Move along Y -->
                    <joint name="cube_{i}_yaw" type="hinge" axis="0 0 1"/>  <!-- Rotate around Z -->
                    <body name="direction_indicator_{i}" pos="0.01 0 0">
                        <geom name="indicator_{i}" type="cylinder" size="0.003 0.00001" rgba="1 1 1 1" euler="0 90 0" density="0"/>
                    </body>            
                </body>"""
            elif shape == 'disc':
                xml += f"""
                <body name="cube_{i}" pos="{x} {y} 0.004">
                    <!-- <geom name="geom_cube_{i}" group="1" type="mesh" mesh="dome_mesh"
                        rgba="{r} {g} {b} 1" density="5000" friction="0.01 0.01 0.01"/> -->
                    <geom name="geom_cube_{i}" group="1" type="cylinder" size="0.01 0.004"
                        rgba="{r} {g} {b} 1" density="5000" friction="0.01 0.01 0.01"/>

                    <joint name="cube_{i}_slide_x" type="slide" axis="1 0 0"/> <!-- Move along X -->
                    <joint name="cube_{i}_slide_y" type="slide" axis="0 1 0"/> <!-- Move along Y -->
                    <joint name="cube_{i}_yaw" type="hinge" axis="0 0 1"/>  <!-- Rotate around Z -->

                    <body name="direction_indicator_{i}" pos="0.01 0 0">
                        <geom name="indicator_{i}" type="cylinder" size="0.002 0.00001" rgba="1 1 1 1" euler="0 90 0" density="0"/>
                    </body>
                </body>"""
            elif shape == 'dome':
                xml += f"""
                <body name="cube_{i}" pos="{x} {y} 0.001">
                    <geom name="geom_cube_{i}" group="1" type="mesh" mesh="disc_dome_mesh"
                        rgba="{r} {g} {b} 1" density="5000" friction="0.01 0.01 0.01"/>

                    <joint name="cube_{i}_slide_x" type="slide" axis="1 0 0"/> <!-- Move along X -->
                    <joint name="cube_{i}_slide_y" type="slide" axis="0 1 0"/> <!-- Move along Y -->
                    <joint name="cube_{i}_yaw" type="hinge" axis="0 0 1"/>  <!-- Rotate around Z -->

                    <body name="direction_indicator_{i}" pos="0.01 0 0.002">
                        <geom name="indicator_{i}" type="cylinder" size="0.002 0.00001" rgba="1 1 1 1" euler="0 60 0" density="0"/>
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
