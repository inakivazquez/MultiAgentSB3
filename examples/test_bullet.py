import pybullet as p
import pybullet_data
import time
import math
import os


def create_goal(position, orientation):
    script_dir = os.path.dirname(__file__)
    goal_path = os.path.join(script_dir, "soc.obj")

    goal_path = "soc.obj"

    scaling_factor = [1, 1, 1]
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=goal_path, meshScale=scaling_factor)

    # Add a collision shape for physics simulation (optional)
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=goal_path, meshScale=scaling_factor)

    # Create a multibody object that combines both visual and collision shapes
    goal_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=orientation)

    return goal_id

# Initialize the simulation
p.connect(p.GUI)

# Set the path to PyBullet's data (optional)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a plane to serve as the ground
plane_id = p.loadURDF("plane.urdf")

create_goal([2, 0, 0.5], p.getQuaternionFromEuler([0, 0, 0]))
create_goal([-2, 0, 0.5], p.getQuaternionFromEuler([0, 0, math.pi]))


# Keep the simulation running to visualize the object
try:
    while True:
        p.stepSimulation()
        time.sleep(1.0/240.0)  # Step the simulation with a specific timestep (240 Hz)

except KeyboardInterrupt:
    # Disconnect the simulation when the user stops it
    p.disconnect()


