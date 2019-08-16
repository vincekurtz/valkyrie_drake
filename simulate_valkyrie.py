#!/usr/bin/env python2

from pydrake.all import *
from helpers import *
from controllers import ValkyrieQPController, ValkyriePDController, ZeroInputController
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
controller = builder.AddSystem(ValkyrieQPController(tree))
builder.Connect(controller.get_output_port(0), robot.get_input_port(0))
builder.Connect(robot.get_output_port(0),controller.get_input_port(0))

# Set contact perameters
contact_params = CompliantContactModelParameters()
contact_params.v_stiction_tolerance = 1e-2     # default 0.01
contact_params.characteristic_radius = 2e-4    # this is a bit stiffer than the default of 2e-4
robot.set_contact_model_parameters(contact_params)

material = CompliantMaterial()
material.set_friction(0.9)
material.set_dissipation(0.32)
material.set_youngs_modulus(1e8)
robot.set_default_compliant_material(material)

diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(True)
context = simulator.get_mutable_context()

# Set initial state
state = context.get_mutable_continuous_state_vector()
initial_state_vec = RPYValkyrieFixedPointState()  # computes [q,qd] for a reasonable starting position
state.SetFromVector(initial_state_vec)

# Use a different integrator to speed up simulation (default is RK3)
#integrator = simulator.get_mutable_integrator()
#integrator.set_fixed_step_mode(True)
#integrator.set_maximum_step_size(0.003)
#integrator.set_target_accuracy(1e-5)
#context.SetAccuracy(1e-5)

integrator = RungeKutta2Integrator(diagram, 2.5e-3, context)  # use 5e-4 to avoid sliding-contact problem 
simulator.reset_integrator(integrator)

# Run the simulation
simulator.Initialize()
simulator.AdvanceTo(2.0)


