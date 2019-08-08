#!/usr/bin/env python2

from pydrake.all import *
from helpers import *
from controllers import ValkyrieController
import numpy as np

# Load the model from a urdf file
robot_description_file = "drake/examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"
robot_urdf = FindResourceOrThrow(robot_description_file)
tree = RigidBodyTree(robot_urdf, FloatingBaseType.kRollPitchYaw)

AddFlatTerrainToWorld(tree, 100, 10)  # add some flat terrain to walk on

# Simulation setup
builder = DiagramBuilder()
lc = DrakeLcm()
robot = builder.AddSystem(RigidBodyPlant(tree,timestep=0))
visualizer = builder.AddSystem(DrakeVisualizer(tree,lc))
builder.Connect(robot.get_output_port(0), visualizer.get_input_port(0))

# Connect the controller
controller = builder.AddSystem(ValkyrieController(tree))
builder.Connect(controller.get_output_port(0), robot.get_input_port(0))
builder.Connect(robot.get_output_port(0),controller.get_input_port(0))

diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)
context = simulator.get_mutable_context()

# Set initial state
state = context.get_mutable_continuous_state_vector()
initial_state_vec = RPYValkyrieFixedPointState()  # computes [q,qd] for a reasonable starting position
state.SetFromVector(initial_state_vec)

# Use a different integrator to speed up simulation (default is RK3)
integrator = RungeKutta2Integrator(diagram, 2e-3, context)
simulator.reset_integrator(integrator)

# Run the simulation
simulator.Initialize()
simulator.AdvanceTo(2.0)
