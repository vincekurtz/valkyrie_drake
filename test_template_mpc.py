#!/usr/bin/env python

from pydrake.all import *
from controllers import ValkyrieASController
from utils.helpers import *

# Load drake model
robot_description_file="drake/examples/valkyrie/urdf/urdf/valkyrie_A_sim_drake_one_neck_dof_wide_ankle_rom.urdf"
robot_urdf = FindResourceOrThrow(robot_description_file)

builder = DiagramBuilder()       # MultiBodyPlant object
scene_graph = builder.AddSystem(SceneGraph())
dt = 1e-3
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
Parser(plant=plant).AddModelFromFile(robot_urdf)
plant.Finalize()

tree = RigidBodyTree(robot_urdf, FloatingBaseType.kRollPitchYaw) # RigidBodyTree object

# create ValkyrieASController Object
c = ValkyrieASController(tree,plant,dt)

# Function to test how well it is possible to satisfy a given linearized CWC constraint
def test_CWC(A_cwc, b_cwc):
    """
    Solve a simple linear program to determine if it is possible to satisfy 
    
    A_cwc*[x_task;u_task] <= b_cwc,

    and if so how well we can satisfy it.
    """
    mp = MathematicalProgram()

    xu = mp.NewContinuousVariables(15,1,"xu")  # [x_task;u_task]
    cost = np.dot(A_cwc,xu) - b_cwc

    mp.AddLinearConstraint(A_cwc, -np.inf*np.ones(b_cwc.shape), b_cwc, xu)
    mp.AddQuadraticErrorCost(np.eye(15),np.zeros((15,1)),xu)

    solver = OsqpSolver()
    res = solver.Solve(mp,None,None)

    if not res.is_success():
        print("Infeasible CWC Constraint!")


# One-shot solve version

# Initial conditions
p_com_init = c.fsm.zmp_trajectory.value(0)  # [x,y]
x_lip = np.asarray([p_com_init[0,0],p_com_init[1,0],0.0,0.0])[np.newaxis].T + 0.005
x_task = np.asarray([p_com_init[0,0],p_com_init[1,0],1,0,0,0,0,0,0])[np.newaxis].T + 0.01

# Perform mpc
u_lip_traj, u_task_traj = c.DoTemplateMPC(0, x_lip, x_task)

n_steps = u_lip_traj.shape[1]
dt = 0.2
sim_time = n_steps*dt

# For storing results
p_zmp_ref = np.zeros((n_steps,2))
p_zmp_lip = np.zeros((n_steps,2))
p_com_lip = np.zeros((n_steps,2))
p_com_task = np.zeros((n_steps,2))
output_err = np.zeros(n_steps)
sim_fcn = np.zeros(n_steps)
x_task_traj = np.zeros((n_steps,9))

# Simulation function V = [x_task;x_lip]'*Q_V*[x_task;x_lip]
Q_V = np.block([
                [ c.M,                -np.dot(c.M,c.P) ],
                [ -np.dot(c.P.T,c.M), np.dot(np.dot(c.P.T,c.M),c.P) ]
              ])
xbar = np.vstack([x_task,x_lip])

V0 = np.dot(np.dot(xbar.T,Q_V),xbar)

# Simulate forward
for i in range(n_steps):
    t = i*dt
    print(t)

    # Extract relevant values
    u_lip = u_lip_traj[:,i][np.newaxis].T
    u_task = u_task_traj[:,i][np.newaxis].T

    # Record plottable values
    p_zmp_ref[i,:] = c.fsm.zmp_trajectory.value(t).flatten()
    p_zmp_lip[i,:] = u_lip.flatten()
    p_com_lip[i,:] = x_lip[0:2].flatten()
    p_com_task[i,:] = x_task[0:2].flatten()
    x_task_traj[i,:] = x_task.flatten()

    output_err[i] = np.linalg.norm( np.dot(c.C_lip,x_lip)-np.dot(c.C_task,x_task) )
   
    xbar = np.vstack([x_task,x_lip])
    sim_fcn[i] = np.sqrt( np.dot(np.dot(xbar.T,Q_V),xbar) )

    # Simulate systems forward in time with Forward Euler
    x_lip = x_lip + dt*(np.dot(c.A_lip,x_lip) + np.dot(c.B_lip,u_lip))
    x_task = x_task + dt*(np.dot(c.A_task,x_task) + np.dot(c.B_task,u_task))


## MPC Recursive solves version
#sim_time = 5
#dt = 0.1
#n_steps = int(sim_time/dt)
#
## For storing results
#p_zmp_ref = np.zeros((n_steps,2))
#p_zmp_lip = np.zeros((n_steps,2))
#p_com_lip = np.zeros((n_steps,2))
#p_com_task = np.zeros((n_steps,2))
#output_err = np.zeros(n_steps)
#sim_fcn = np.zeros(n_steps)
#
## Initial conditions
#p_com_init = c.fsm.zmp_trajectory.value(0)  # [x,y]
#x_lip = np.asarray([p_com_init[0,0],p_com_init[1,0],0.0,0.0])[np.newaxis].T + 0.005
#x_task = np.asarray([p_com_init[0,0],p_com_init[1,0],1,0,0,0,0,0,0])[np.newaxis].T + 0.01
#
#for i in range(n_steps):
#    t = i*dt
#    print(t)
#
#    # Perform mpc
#    u_lip_traj, u_task_traj = c.DoTemplateMPC(t, x_lip, x_task)
#
#    # Extract relevant values
#    u_lip = u_lip_traj[:,0][np.newaxis].T
#    u_task = u_task_traj[:,0][np.newaxis].T
#
#    # Record plottable values
#    p_zmp_ref[i,:] = c.fsm.zmp_trajectory.value(t).flatten()
#    p_zmp_lip[i,:] = u_lip.flatten()
#    p_com_lip[i,:] = x_lip[0:2].flatten()
#    p_com_task[i,:] = x_task[0:2].flatten()
#
#    output_err[i] = np.linalg.norm( np.dot(c.C_lip,x_lip)-np.dot(c.C_task,x_task) )
#   
#    x_Px = x_task - np.dot(c.P,x_lip)
#    sim_fcn[i] = np.sqrt( np.dot( np.dot(x_Px.T,c.M), x_Px) )
#
#    # Simulate systems forward in time with Forward Euler
#    x_lip = x_lip + dt*(np.dot(c.A_lip,x_lip) + np.dot(c.B_lip,u_lip))
#    x_task = x_task + dt*(np.dot(c.A_task,x_task) + np.dot(c.B_task,u_task))

###############################################
# Plot Results
###############################################

# Target ZMP trajectory
plt.plot(p_zmp_ref[:,0],p_zmp_ref[:,1], label="Target ZMP")

# LIPM ZMP trajectory
plt.plot(p_zmp_lip[:,0], p_zmp_lip[:,1], label="LIP ZMP")

# LIPM CoM trajectory
plt.plot(p_com_lip[:,0], p_com_lip[:,1], label="LIP CoM")

# Actual (task-space) CoM trajectory
plt.plot(p_com_task[:,0], p_com_task[:,1], label="Task-space CoM")

plt.xlabel("x position")
plt.ylabel("y position")
plt.xlim(-0.2,0.8)
plt.ylim(-0.2,0.2)

plt.legend()

# Output error and simulation function
plt.figure()
plt.plot(np.arange(0,sim_time,dt),output_err, label="Output Error")
plt.plot(np.arange(0,sim_time,dt),sim_fcn, label="Simulation Function")
plt.xlabel("Time")

plt.legend()
#
## Plot of task-space trajectory
#plt.figure()
#plt.plot(x_task_traj)
#plt.legend(["px", "py", "pz", "kx", "ky", "kz", "lx", "ly", "lz"])

plt.show()
