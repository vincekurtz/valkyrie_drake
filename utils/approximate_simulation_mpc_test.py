#!/usr/bin/env python

#
# Script to test simultaneous MPC planning and tracking using approximate simulation
#

import numpy as np
from pydrake.all import *
from helpers import *
import matplotlib.pyplot as plt

###############################################
# Generate reference ZMP trajectory
###############################################

# Specify break points for piecesise linear interposlation
zmp_ref = np.asarray([[0.0,  0.0],
                      [0.0,  0.14],
                      [0.0,  0.14],
                      [0.2,-0.14],
                      [0.2,-0.14],
                      [0.4,  0.14],
                      [0.4,  0.14],
                      [0.6,-0.14],
                      [0.6,-0.14],
                      [0.8,  0.14],
                      [0.8,  0.14],
                      [0.8,  0.0],
                      [0.8,  0.0]]).T

# Specify break times for piecewise linear interpolation
step_time = 0.3
break_times = np.asarray([[i*step_time] for i in range(zmp_ref.shape[1])])

# Perform polynomial interpolation
zmp_trajectory = PiecewisePolynomial.FirstOrderHold(break_times, zmp_ref)

###############################################
# System Dynamics definitions
###############################################

# Physical system parameters
h = 0.967
g = 9.81
m = 136.0
omega = np.sqrt(g/h)

# LIPM dynamics
A_lip = np.zeros((4,4))
A_lip[0:2,2:4] = np.eye(2)
A_lip[2:4,0:2] = omega**2*np.eye(2)

B_lip = np.zeros((4,2))
B_lip[2:4,:] = -omega**2*np.eye(2)

C_lip = np.zeros((9,4))       # Output for approximate simulation
C_lip[0:2,0:2] = np.eye(2)
C_lip[6:8,2:4] = m*np.eye(2)

C_lip_zmp = np.zeros((2,4))   # Output for ZMP planning = CoM acceleration
C_lip_zmp[0:2,0:2] = omega**2*np.eye(2)
D_lip_zmp = -omega**2*np.eye(2)

# Task-space (centroidal) dynamics
A_task = np.zeros((9,9))
A_task[0:3,6:9] = 1/m*np.eye(3)

B_task = np.zeros((9,6))
B_task[3:9,:] = np.eye(6)

C_task = np.eye(9)

###############################################
# Interface Definition
###############################################

###############################################
# MPC formulation
###############################################

def perform_template_mpc(t, x_lip_init, x_task_init):
    # Prediction horizon and sampling times
    N = 50
    dt = 0.05

    # MPC parameters
    R_mpc = 100*np.eye(2)   # ZMP tracking penalty
    Q_mpc = np.eye(2)       # Penalty on CoM velocity
    Qf_mpc = 10*np.eye(2)      # Final penalty on CoM velocity

    # Set up a Drake MathematicalProgram
    mp = MathematicalProgram()

    # Create optimization variables
    x_lip = mp.NewContinuousVariables(4,N,"x_lip")    # position and velocity of CoM in plane
    u_lip = mp.NewContinuousVariables(2,N-1,"u_lip")  # center of pressure position on the ground plane

    x_task = mp.NewContinuousVariables(9,N,"x_task")    # CoM position and centroidal momentum
    u_task = mp.NewContinuousVariables(6,N-1,"u_task")  # Spatial force on CoM (centroidal momentum dot)
        

    # Initial condition constraints
    mp.AddLinearEqualityConstraint(np.eye(4), x_lip_init, x_lip[:,0])
    mp.AddLinearEqualityConstraint(np.eye(9), x_task_init, x_task[:,0])

    for i in range(N-1):

        # Add Running Costs
        zmp_des = zmp_trajectory.value(t+dt*i)
        mp.AddQuadraticErrorCost(R_mpc,zmp_des,u_lip[:,i])            # regulate ZMP to track nominal
        mp.AddQuadraticErrorCost(Q_mpc,np.zeros((2,1)),x_lip[2:4,i])  # Penalize CoM velocity

        # Add dynamic constraints for the LIPM
        AddForwardEulerDynamicsConstraint(mp, A_lip, B_lip, 
                                          x_lip[:,i], u_lip[:,i], x_lip[:,i+1],
                                          dt)

        # Add dynamic constraints for the task space
        AddForwardEulerDynamicsConstraint(mp, A_task, B_task, 
                                          x_task[:,i], u_task[:,i], x_task[:,i+1],
                                          dt)

        ## Add interface constraint
        #A_interface = np.hstack([self.R, (self.Q-np.dot(self.K,self.P)), self.K, -np.eye(6)])
        #x_interface = np.hstack([u_lip[:,i],x_lip[:,i],x_task[:,i],u_task[:,i]])[np.newaxis].T
        #mp.AddLinearEqualityConstraint(A_interface, np.zeros((6,1)), x_interface)


    # Add terminal cost
    mp.AddQuadraticErrorCost(Q_mpc,np.zeros((2,1)),x_lip[2:4,N-1])  # Penalize CoM accelrations
        
    # Solve the QP
    solver = OsqpSolver()
    res = solver.Solve(mp,None,None)

    x_lip_traj = res.GetSolution(x_lip)
    u_lip_traj = res.GetSolution(u_lip)
    x_task_traj = res.GetSolution(x_task)
    u_task_traj = res.GetSolution(u_task)
    
    return x_lip_traj, u_lip_traj, x_task_traj, u_task_traj

###############################################
# Run simulation
###############################################

sim_time = step_time*zmp_ref.shape[1]
dt = 0.05
n_steps = int(sim_time/dt)

# For storing results
p_zmp_ref = np.zeros((n_steps,2))
p_zmp_lip = np.zeros((n_steps,2))
p_com_lip = np.zeros((n_steps,2))

# Initial conditions
x_lip = np.zeros((4,1))
x_task = np.zeros((9,1))

for i in range(n_steps):
    t = i*dt

    # Perform mpc
    x_lip_traj, u_lip_traj, x_task_traj, u_task_traj = perform_template_mpc(t, x_lip, x_task)

    # Extract control inputs
    u_lip = u_lip_traj[:,0][np.newaxis].T
    u_task = u_task_traj[:,0][np.newaxis].T

    # Record plottable values
    p_zmp_ref[i,:] = zmp_trajectory.value(t).flatten()
    p_zmp_lip[i,:] = u_lip_traj[:,0].flatten()
    p_com_lip[i,:] = x_lip_traj[0:2,0].flatten()

    # Simulate systems forward in time with Forward Euler
    x_lip = x_lip + dt*(np.dot(A_lip,x_lip) + np.dot(B_lip,u_lip))
    x_task = x_task + dt*(np.dot(A_task,x_task) + np.dot(B_task,u_task))


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



plt.legend()
plt.show()

