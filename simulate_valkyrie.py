#!/usr/bin/env python2

from pydrake.all import *
from utils.helpers import ValkyrieFixedPointState
from utils.disturbance_system import DisturbanceSystem
from controllers import ValkyriePDController, ValkyrieQPController, ValkyrieASController
import numpy as np
import matplotlib.pyplot as plt
import sys

######################################################################
# Simulation Parameters
######################################################################

# Specify whether the controller should assume an incorrect model
use_incorrect_model = True
model_num = sys.argv[1]    # incorrect model in [0,10] to use if use_incorrect_model is True

# Specify whether to add a random lateral push
add_random_push = False
push_seed = None

# Specify whether to add uneven terrain
add_uneven_terrain = False
terrain_seed = None   # random seed used to generate uneven terrain

# Specify control method: "AS" (our proposed approach) or "QP" (standard QP)
control_method = "QP"

# Specify total simulation time in seconds
sim_time = 5.0

# Specify whether to make plots at the end
make_plots = False

# Specify whether to include state estimation noise on floating base
use_estimation_noise = False
sigma_p = 5.7      # position error std deviation in mm
sigma_r = 0.5      # rotation error std deviation in degrees
sigma_v = 18.5     # velocity error std deviation in mm/s

######################################################################

# Specify (potentially different) models for the simulator and for the controller
assumed_robot_description_file = "drake/examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"
if use_incorrect_model:
    true_robot_description_file = "drake/examples/valkyrie/urdf/urdf/valkyrie_modified_%s.urdf" % model_num
else:
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
        static_friction = 1.0,
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

#plant.set_stiction_tolerance(0.001)

if add_uneven_terrain:
# Add uneven terrain to the world, by placing a bunch of block randomly
    terrain_file = FindResourceOrThrow("drake/manipulation/models/ycb/sdf/block.sdf")
    np.random.seed(terrain_seed)  # fix random seed for reproducability
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
plant.set_penetration_allowance(0.002)
assert plant.geometry_source_is_registered()

if add_random_push:
    # Set up an external force
    np.random.seed(push_seed)
    time = np.random.uniform(low=1.0,high=3.5)
    magnitude = np.random.uniform(low=400,high=600)
    direction = np.random.choice([-1,1])

    disturbance_sys = builder.AddSystem(DisturbanceSystem(plant,
                                                          "torso",                     # body to apply to
                                                          np.asarray([0,0,0,0,direction*magnitude,0]),  # wrench to apply
                                                          time,                         # time
                                                          0.05))                        # duration
    builder.Connect(
            disturbance_sys.get_output_port(0),
            plant.get_applied_spatial_force_input_port())

# Set up the Scene Graph
builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))

# Set up state estimation noise
if use_estimation_noise:
    estimation_noise = (sigma_p/1000,       # convert mm to m
                        sigma_r*np.pi/180,  # convert degrees to radians
                        sigma_v/1000)       # convert mm/s to m/s
else:
    estimation_noise = None

# Set up a controller
if control_method == "AS":
    ctrl = ValkyrieASController(tree,plant,dt,estimation_noise)
elif control_method == "QP":
    ctrl = ValkyrieQPController(tree,plant,estimation_noise)
else:
    raise(ValueError("invalid control method %s" % control_method))

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
try:
    simulator.AdvanceTo(sim_time)
except:
    sys.exit(1)


if make_plots:
    # make some plots of the results

    # Plot of y1 vs y2
    #plt.figure()
    #plt.subplot(3,1,1)
    #plt.plot(ctrl.t, ctrl.y1[0,1:], label="Actual ($\mathbf{y}_1$)", linewidth='2')
    #plt.plot(ctrl.t, ctrl.y2[0,1:], "--", label="CoM Model ($\mathbf{y}_2$)", linewidth='2')
    #plt.ylabel("CoM x position (m)")
    ##plt.ylim(-2,2)
    #plt.legend()
    ##plt.title("CoM Position Tracking")

    #plt.subplot(3,1,2)
    #plt.plot(ctrl.t, ctrl.y1[1,1:], label="Actual ($\mathbf{x}_1$)", linewidth='2')
    #plt.plot(ctrl.t, ctrl.y2[1,1:], "--", label="CoM Model ($\mathbf{x}_2$)", linewidth='2')
    #plt.ylabel("CoM y position (m)")
    ##plt.ylim(-2,2)

    #plt.subplot(3,1,3)
    #plt.plot(ctrl.t, ctrl.y1[2,1:], label="Actual ($\mathbf{x}_1$)", linewidth='2')
    #plt.plot(ctrl.t, ctrl.y2[2,1:], "--", label="CoM Model ($\mathbf{x}_2$)", linewidth='2')
    ##plt.hlines(ctrl.fsm.x_com_init[2],0,ctrl.t[-1],linestyles="dashdot",linewidth='2',label="LIP Model ($\mathbf{x}_3$)")
    ##plt.ylim(0.92,1.07)
    #plt.ylabel("CoM z position (m)")
    #plt.xlabel("Time (s)")
    ##plt.legend(loc=4)
    ##plt.ylim(-2,2)


    # Plot of simulation function vs error
    #plt.plot(ctrl.t, ctrl.V, label="Simulation Function", linewidth='2')
    #plt.ylabel("Simulation Function / Output Error")
    #plt.plot(ctrl.t, ctrl.err, "--", label="Output Error", color='green', linewidth='2')
    #plt.legend()

    # Plot torque profile
    plt.figure()
    plt.plot(ctrl.t, ctrl.tau[:,1:].T, linewidth='2')
    plt.ylabel("Joint Torques")
    plt.xlabel("Time (s)")
    #plt.title("Torque Profile")

    tau_squared = []
    for i in range(ctrl.tau.shape[1]):
        tau_squared.append(np.dot(ctrl.tau[:,i].T,ctrl.tau[:,i]))

    plt.figure()
    plt.plot(ctrl.t,tau_squared[1:],linewidth='2')

    # Compute integral of torques squared
    #TT = 0
    #for i in range(ctrl.tau.shape[1]):
    #    TT += np.dot(ctrl.tau[:,i].T,ctrl.tau[:,i])*dt
    #print("Integral of torques squared: %s" % TT)

    # Compute approximate error bound
    #eps = np.asarray(ctrl.epsilon)[1:]
    #print("Approximate Error Bound: %s" % np.max(eps))



    plt.show()

