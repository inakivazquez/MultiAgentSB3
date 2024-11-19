import pybullet as p
import math
import pybullet_data
import time

def generate_vectors(n, radius, z):
    vectors = []
    angle_increment = 2 * math.pi / n
    for i in range(n):
        angle = i * angle_increment
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vectors.append((x, y, z))
    return vectors

def detect_objects(vectors):
    start_positions = [(0, 0, 0)] * len(vectors)  # Starting from origin for each vector
    end_positions = vectors  # End positions are the given vectors

    # Perform ray tests
    ray_results = p.rayTestBatch(start_positions, end_positions)

    results = []
    for start, end, result in zip(start_positions, end_positions, ray_results):
        object_id = result[0]
        hit_position = result[3]
        distance = math.sqrt((hit_position[0] - start[0])**2 + 
                             (hit_position[1] - start[1])**2 + 
                             (hit_position[2] - start[2])**2)
        results.append((object_id, distance))

        # Draw debug line for each ray
        p.addUserDebugLine(start, end, lineColorRGB=[1, 0, 0], lifeTime=0)  # Red lines

    return results

# Example usage
if __name__ == "__main__":
    # Connect to PyBullet (assumes GUI mode for visualization)
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)

    # Load some objects into the simulation for testing
    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [1, 1, 0.5])
    p.loadURDF("cube.urdf", [2, 3, 0.5])

    # Generate vectors
    n_vectors = 36
    radius = 10  # Arbitrary radius for testing
    z_plane = 0.5

    vectors = generate_vectors(n_vectors, radius, z_plane)

    # Detect objects
    results = detect_objects(vectors)

    # Print the results
    for i, (object_id, distance) in enumerate(results):
        if object_id != -1:
           print(f"Vector {i}: Object ID = {object_id}, Distance = {distance}")

    while True:
        p.stepSimulation()
        time.sleep(1./240.)
    # Disconnect from PyBullet
    p.disconnect()
