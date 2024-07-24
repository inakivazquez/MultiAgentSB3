import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a plane and a cube
p.loadURDF("plane.urdf")
cube_id = p.loadURDF("cube.urdf", basePosition=[0, 0, 1])

# Apply a force to the cube
force = [20, 10, 0]  # Example force vector
position = [0, 0, 1]  # Position where the force is applied
position = p.getBasePositionAndOrientation(cube_id)[0]  # Get the position of the cube
p.applyExternalForce(cube_id, linkIndex=-1, forceObj=force, posObj=position, flags=p.WORLD_FRAME)

# Step the simulation to apply the force
for _ in range(10000):  # Run simulation for a few steps
    p.stepSimulation()
    time.sleep(1/240)  # Sleep for the duration of one timestep

    # Get the base velocity of the cube
    linear_velocity, angular_velocity = p.getBaseVelocity(cube_id)
    
    # Print the velocities
    print(f"Linear Velocity (world frame): {linear_velocity}")
    print(f"Angular Velocity (world frame): {angular_velocity}")

# Cleanup
p.disconnect()
