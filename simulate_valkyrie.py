#!/usr/bin/env python2

from pydrake.all import *
import numpy as np

# Load a model from a urdf file

#robot_description_file = "drake/manipulation/models/jaco_description/urdf/j2n6s300.urdf"
robot_description_file = "drake/examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"
robot_urdf = FindResourceOrThrow(robot_description_file)
tree = RigidBodyTree(robot_urdf, FloatingBaseType.kFixed)

#AddFlatTerrainToWorld(tree, 100, 10)  # add some flat terrain to walk on

# See some model parameters
print(tree.get_num_positions())
print(tree.get_num_velocities())
print(tree.get_num_actuators())
print(tree.get_num_bodies())
print("")

# Get equations of motion for given q, qd
q = np.zeros(tree.get_num_positions())
qd = np.zeros(tree.get_num_velocities())
cache = tree.doKinematics(q,qd)

bodies = [tree.get_body(j) for j in range(tree.get_num_bodies())]
no_wrench = { body : np.zeros(6) for body in bodies}

C = tree.dynamicsBiasTerm(cache, no_wrench)  # correolis and centripedal terms
H = tree.massMatrix(cache)                   # mass matrix

print(C.shape)
print(H.shape)

# Centroidal momentum quantities
A = tree.centroidalMomentumMatrix(cache)
Ad_qd = tree.centroidalMomentumMatrixDotTimesV(cache)
print(A.shape)
print(Ad_qd.shape)

# Simulation: start the visualizer in another terminal (bazel-bin/tools/drake-visualizer)
builder = DiagramBuilder()

lc = DrakeLcm()
vis = DrakeVisualizer(tree, lc)
robot = builder.AddSystem(RigidBodyPlant(tree))
publisher = builder.AddSystem(vis)
builder.Connect(robot.get_output_port(0), publisher.get_input_port(0))
force = builder.AddSystem(ConstantVectorSource(np.zeros(tree.get_num_actuators())))
builder.Connect(force.get_output_port(0), robot.get_input_port(0))

diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)
context = simulator.get_mutable_context()
state = context.get_mutable_continuous_state_vector()
state.SetFromVector(np.zeros(tree.get_num_positions()+tree.get_num_velocities())+0.5)

integrator = simulator.get_mutable_integrator()  # set some integration parameters to 
integrator.set_fixed_step_mode(True)             # speed up simulation
integrator.set_maximum_step_size(0.004)
integrator.set_target_accuracy(1e-2)
context.SetAccuracy(1e-2)

simulator.Initialize()
simulator.AdvanceTo(1)
