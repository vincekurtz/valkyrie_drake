#!/usr/bin/env python2

from pydrake.all import *
from helpers import RPYValkyrieFixedPointState, RPYValkyrieFixedPointTorque
from controllers import ValkyrieController
import numpy as np

# Load the model from a urdf file
robot_description_file = "drake/examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"
robot_urdf = FindResourceOrThrow(robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
robot = builder.AddSystem(MultibodyPlant(time_step=0))   # 0 for continuous, dt for discrete
robot.RegisterAsSourceForSceneGraph(scene_graph)
Parser(plant=robot).AddModelFromFile(robot_urdf)
robot.Finalize()
assert robot.geometry_source_is_registered()

#tree = RigidBodyTree(robot_urdf, FloatingBaseType.kRollPitchYaw)

#AddFlatTerrainToWorld(tree, 100, 10)  # add some flat terrain to walk on

# Set up the Scene Graph
builder.Connect(
        scene_graph.get_query_output_port(),
        robot.get_geometry_query_input_port())
builder.Connect(
        robot.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(robot.get_source_id()))

# Connect the visualizer
ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
diagram = builder.Build()

diagram_context = diagram.CreateDefaultContext()
robot_context = diagram.GetMutableSubsystemContext(robot, diagram_context)

# Send fixed commands of zero to the joints
zero_cmd = np.zeros(robot.num_actuators())
robot_context.FixInputPort(
        robot.get_actuation_input_port().get_index(), 
        np.zeros(30))


# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

# Set initial state
#state = context.get_mutable_continuous_state_vector()
#initial_state_vec = RPYValkyrieFixedPointState()  # computes [q,qd] for a reasonable starting position
#state.SetFromVector(initial_state_vec)

# Use a different integrator to speed up simulation (default is RK3)
#context = simulator.get_mutable_context()
#integrator = RungeKutta2Integrator(diagram, 1e-3, context)
#simulator.reset_integrator(integrator)

# Run the simulation
simulator.Initialize()
simulator.AdvanceTo(1.0)
