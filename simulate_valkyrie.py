#!/usr/bin/env python2

from pydrake.all import *
from utils.helpers import ValkyrieFixedPointState
from controllers import ValkyriePDController, ValkyrieQPController, ValkyrieASController
import numpy as np

# Load the valkyrie model from a urdf file
robot_description_file = "drake/examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"
robot_urdf = FindResourceOrThrow(robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
dt = 1e-2
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
Parser(plant=plant).AddModelFromFile(robot_urdf)

# Use the (admittedly depreciated) RigidBodyTree interface for dynamics
# calculations, since python bindings for MultibodyPlant dynamics don't seem to 
# exist yet. This should allow us to keep using MBP for the simulation, which seems to work a lot better.
tree = RigidBodyTree(robot_urdf, FloatingBaseType.kRollPitchYaw)

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

plant.Finalize()
assert plant.geometry_source_is_registered()

# Set up the Scene Graph
builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))

# Set up a controller
#controller = builder.AddSystem(ValkyrieASController(tree,plant,dt))
controller = builder.AddSystem(ValkyrieQPController(tree,plant))
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
simulator.AdvanceTo(10.0)
