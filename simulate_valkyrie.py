#!/usr/bin/env python2

from pydrake.all import *
from utils.helpers import ValkyrieFixedPointState
from utils.disturbance_system import DisturbanceSystem
from controllers import ValkyriePDController, ValkyrieQPController, ValkyrieASController
import numpy as np
import matplotlib.pyplot as plt

# Specify (potentially different) models for the simulator and for the controller
assumed_robot_description_file = "drake/examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"
true_robot_description_file = "drake/examples/valkyrie/urdf/urdf/valkyrie_modified.urdf"
true_robot_description_file = assumed_robot_description_file

# Load the valkyrie model from a urdf file
robot_urdf = FindResourceOrThrow(true_robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
dt = 5e-3
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
Parser(plant=plant).AddModelFromFile(robot_urdf)

# Use the (admittedly depreciated) RigidBodyTree interface for dynamics
# calculations, since python bindings for MultibodyPlant dynamics don't seem to 
# exist yet. This should allow us to keep using MBP for the simulation, which seems to work a lot better.
tree_robot_urdf = FindResourceOrThrow(assumed_robot_description_file)
tree = RigidBodyTree(tree_robot_urdf, FloatingBaseType.kRollPitchYaw)

# Add a flat ground with friction
X_BG = RigidTransform()
surface_friction = CoulombFriction(
        static_friction = 0.7,
        dynamic_friction = 0.1)
plant.RegisterCollisionGeometry(
        plant.world_body(),      # the body for which this object is registered
        X_BG,                    # The fixed pose of the geometry frame G in the body frame B
        HalfSpace(),             # Defines the geometry of the object
        "ground_collision",      # A name
        surface_friction)        # Coulomb friction coefficients
plant.RegisterVisualGeometry(
        plant.world_body(),
        X_BG,
        HalfSpace(),
        "ground_visual",
        np.array([0.5,0.5,0.5,0.0]))    # Color set to be completely transparent

# Add uneven terrain to the world, by placing a bunch of block randomly
terrain_file = FindResourceOrThrow("drake/manipulation/models/ycb/sdf/block.sdf")
np.random.seed(10)  # fix random seed for reproducability
for i in range(15):
    # load a block from a urdf file
    terrain = Parser(plant=plant).AddModelFromFile(terrain_file, "terrain_block_%s" % i)

    # Generate random RPY and position values
    x_min = 0.25; x_max = 1.5
    y_min = -0.3; y_max = 0.3
    z_min = -0.015; z_max = 0.002

    r_min = -np.pi/20; r_max = np.pi/20
    p_min = -np.pi/20; p_max = np.pi/20
    yy_min = -np.pi; yy_max = np.pi

    x = np.random.uniform(low=x_min,high=x_max)
    y = np.random.uniform(low=y_min,high=y_max)
    z = np.random.uniform(low=z_min,high=z_max)

    r = np.random.uniform(low=r_min,high=r_max)
    r = 0
    p = np.random.uniform(low=p_min,high=p_max)
    p = 0
    yy = np.random.uniform(low=yy_min,high=yy_max)

    # weld the block to the world at this pose
    R = RollPitchYaw(np.asarray([r,p,yy])).ToRotationMatrix()
    p = np.array([x,y,z]).reshape(3,1)
    X = RigidTransform(R,p)
    plant.WeldFrames(plant.world_frame(),plant.GetFrameByName("base_link",terrain),X)


plant.Finalize()
assert plant.geometry_source_is_registered()

# Set up an external force
#disturbance_sys = builder.AddSystem(DisturbanceSystem(plant,
#                                                      "torso",                     # body to apply to
#                                                      np.asarray([0,0,0,0,-200,0]),  # wrench to apply
#                                                      1.1,                         # time
#                                                      0.05))                        # duration
#builder.Connect(
#        disturbance_sys.get_output_port(0),
#        plant.get_applied_spatial_force_input_port())

# Set up the Scene Graph
builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))

# Set up a controller
#ctrl = ValkyrieASController(tree,plant,dt)
ctrl = ValkyrieQPController(tree,plant)
controller = builder.AddSystem(ctrl)
builder.Connect(
        plant.get_state_output_port(),
        controller.get_input_port(0))
builder.Connect(
        controller.get_output_port(0),
        plant.get_actuation_input_port(plant.GetModelInstanceByName('valkyrie')))

# Set up the Visualizer
ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant)

# Compile the diagram: no adding control blocks from here on out
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.5)
simulator.set_publish_every_time_step(False)

# Set initial state
state = plant_context.get_mutable_discrete_state_vector()
initial_state_vec = ValkyrieFixedPointState()  # computes [q,v] for a reasonable starting position
state.SetFromVector(initial_state_vec)

# Run the simulation
simulator.Initialize()
simulator.AdvanceTo(15.0)

####################################################################
# Make some plots
####################################################################

# Plot of y1 vs y2
plt.figure()
plt.subplot(3,1,1)
plt.plot(ctrl.t, ctrl.y1[0,1:], label="actual", linewidth='2')
plt.plot(ctrl.t, ctrl.y2[0,1:], "--", label="desired", linewidth='2')
plt.ylabel("x")
#plt.ylim(-2,2)
plt.legend()
plt.title("CoM Position Tracking")

plt.subplot(3,1,2)
plt.plot(ctrl.t, ctrl.y1[1,1:], label="actual", linewidth='2')
plt.plot(ctrl.t, ctrl.y2[1,1:], "--", label="desired", linewidth='2')
plt.ylabel("y")
#plt.ylim(-2,2)

plt.subplot(3,1,3)
plt.plot(ctrl.t, ctrl.y1[2,1:], label="actual", linewidth='2')
plt.plot(ctrl.t, ctrl.y2[2,1:], "--", label="desired", linewidth='2')
plt.ylabel("z")
plt.xlabel("time")
#plt.ylim(-2,2)


# Plot of simulation function vs error
plt.figure()
plt.subplot(2,1,1)
plt.plot(ctrl.t, ctrl.V, label="Simulation Function", linewidth='2')
plt.ylabel("Simulation Function")
plt.title("Simulation Fcn vs. Output Error")

plt.subplot(2,1,2)
plt.plot(ctrl.t, ctrl.err, label="Output Error", color='green', linewidth='2')
plt.ylabel("Output Error")
plt.xlabel("time")

# Plot torque profile
plt.figure()
plt.plot(ctrl.t, ctrl.tau[:,1:].T, linewidth='2')
plt.ylabel("Joint Torques")
plt.xlabel("time")
plt.title("Torque Profile")


plt.show()

