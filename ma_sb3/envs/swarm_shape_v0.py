from ma_sb3.envs.base_swarm_env import BaseSwarmEnv
import numpy as np
import mujoco

class SwarmShapeEnv(BaseSwarmEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asset_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "asset")

    def evaluate_env_state(self):
        truncated = False
        rewards = {}
        infos = {}   
        terminated = False

        # Initialize rewards dictionary
        for agent_id in self.agents:
            rewards[agent_id] = 0

        # Store the x, y coordinates of all agents in a numpy array
        agent_positions = np.array([
            self.data.xpos[self.mujoco_cube_ids[agent_id]][:2] 
            for agent_id in self.agents
        ])

        asset_distance_required = 0.2
        # Compute the circularity score of the agent positions
        circularity, distance_scores, angular_scores = self.circle_fit_score(agent_positions, 0, 0, asset_distance_required)

        circularity_required = 0.98
        average_distance_score = np.mean(distance_scores)
        for i,agent_id in enumerate(self.agents):
            if circularity >= circularity_required and average_distance_score > 0.90 and False:
                rewards[agent_id] += 100  # Reward agents with the circularity score
                #rewards[agent_id] -= (1 - np.mean(distance_scores))*100  # Penalty average distance from the circle
                print(f"Agent {agent_id} circularity: {circularity}, distance: {distance_scores[i]}")
            else:
                # Penalty based on movements
                mujoco_body_id = self.mujoco_cube_ids[agent_id]
                if self.active_movements.get(mujoco_body_id) is not None: # For the first state
                    rewards[agent_id] -= self.active_movements[mujoco_body_id]['distance_done']  # Penalty based on distance done
                    rewards[agent_id] -= self.active_movements[mujoco_body_id]['rotation_done'] / 10  # Penalty based on rotation done
                    pass
                # Reward based on shape scores
                rewards[agent_id] += distance_scores[i] / 10  # Reward agents with the individual distance score
                #rewards[agent_id] += angular_scores[i] / 10  # Reward agents with the individual angular score
                if distance_scores[i] > 0.90:
                    rewards[agent_id] += 4 * circularity / 10.0   # Reward agents with the collective circularity score
                #rewards[agent_id] -= 0.01  # Step penalty
                #print(f"Agent {agent_id} circularity: {circularity}, distance: {distance_scores[i]}, angular: {angular_scores[i]}")

        if circularity >= circularity_required and average_distance_score > 0.90:
            #terminated = True
            print(f"Achieved circularity: {circularity}!")

        return rewards, terminated, truncated, infos

    def get_observation(self, agent_id):
        mujoco_cube_id = self.mujoco_cube_ids[agent_id]
        detected_body_ids, normalized_distances = self.perform_raycast(mujoco_cube_id)
        ray_obs = np.zeros((self.nrays, 3+self.individual_comm_items), dtype=np.float32)
        for i, detected_body_id in enumerate(detected_body_ids):
            if detected_body_id == self.asset_id:
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

        xml += f"""       
            <body name="asset" pos="0 0 0.1">
                <joint type="free"/> 
                <geom name="geom_asset" group="1" type="cylinder" size="0.05 0.05" rgba="0.0 0.8 0.0 0.4" density="5000"/>
            </body> 
            </worldbody>
        </mujoco>"""
        
        return xml
