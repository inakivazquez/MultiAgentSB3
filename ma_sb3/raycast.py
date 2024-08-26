import math
import pybullet as p
import numpy as np

def raycast_horizontal_detect(source_pos, source_angle_z, n_raycasts=7, covering_angle=2*math.pi, vision_length=10, draw_lines=True):

    # Generate vectors
    vectors = []
    
    # Front direction
    angle_increment = covering_angle / n_raycasts

    for i in range(n_raycasts):
        # Starting raycast around source_angle_z direction
        angle = source_angle_z + (i - n_raycasts//2 ) * angle_increment
        x = vision_length * math.cos(angle)
        y = vision_length * math.sin(angle)
        vectors.append((x, y, source_pos[2]))

    start_positions = [(source_pos[0], source_pos[1], source_pos[2])] * len(vectors)  # Starting from origin for each vector
    end_positions = vectors  # End positions are the generated vectors

    # Perform ray tests
    ray_results = p.rayTestBatch(start_positions, end_positions)

    results = np.array([]).reshape(0, 2)
    for start, end, result in zip(start_positions, end_positions, ray_results):
        object_id = result[0]
        hit_position = result[3]
        distance = math.sqrt((hit_position[0] - start[0])**2 + 
                                (hit_position[1] - start[1])**2 + 
                                (hit_position[2] - start[2])**2)
        
        results = np.append(results, [[object_id , distance]], axis=0)

        # Draw debug line for each ray
        if draw_lines:
            p.addUserDebugLine(start, end, lineColorRGB=[1, 0, 0], lifeTime=0)  # Red lines

    return results
    
