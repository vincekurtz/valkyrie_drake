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

C_lip = np.zeros((9,4))
C_lip[0:2,0:2] = np.eye(2)
C_lip[6:8,2:4] = m*np.eye(2)

# Task-space (centroidal) dynamics
A_task = np.zeros((9,9))
A_task[0:3,6:9] = 1/m*np.eye(3)

B_task = np.zeros((9,6))
B_task[3:9,:] = np.eye(6)

C_task = np.eye(9)

###############################################
# Interface Definition
###############################################

interface_mp = MathematicalProgram()

# Find M, K by solving an SDP
lmbda = 0.01
Mbar = interface_mp.NewSymmetricContinuousVariables(9,"Mbar")
Kbar = interface_mp.NewContinuousVariables(6,9,"Kbar")

sdcon_1 = np.vstack([
          np.hstack([ Mbar,                np.dot(Mbar,C_task.T) ]),
          np.hstack([ np.dot(C_task,Mbar), np.eye(9)             ])])

sdcon_2 = -(np.dot(Mbar, A_task.T) + np.dot(A_task, Mbar) + np.dot(Kbar.T,B_task.T) + np.dot(B_task,Kbar) + 2*lmbda*Mbar)

# Add semidefinite constraints with some epsilon for numerical stability
interface_mp.AddPositiveSemidefiniteConstraint(sdcon_1 - 1e-5*np.eye(sdcon_1.shape[0]))
interface_mp.AddPositiveSemidefiniteConstraint(sdcon_2 - 1e-5*np.eye(sdcon_2.shape[0]))
interface_mp.AddPositiveSemidefiniteConstraint(Mbar-1e-3*np.eye(9))

interface_mp.AddCost(-np.trace(Mbar))                # incentivize a tight error bound 
interface_mp.AddCost(np.trace(np.dot(Kbar.T,Kbar)))  # penalize high control gains

result = Solve(interface_mp)

assert result.is_success(), "Interface SDP infeasible"
M = np.linalg.inv(result.GetSolution(Mbar))
K = np.dot(result.GetSolution(Kbar),M)

# Choose P, Q, and R by hand
P = C_lip

Q = np.zeros((6,4))
Q[3:5,0:2] = omega**2*m*np.eye(2)

R = np.zeros((6,2))
R[3:5] = -m*omega**2*np.eye(2)

# Double check the results
assert is_pos_def(M) , "M is not positive definite."
assert is_pos_def(-A_task-np.dot(B_task,K)) , "A+BK is not Hurwitz"

assert is_pos_def(M - np.dot(C_task.T,C_task)) , "Failed test M >= C'C"
assert is_pos_def(-2*lmbda*M \
                  - np.dot((A_task+np.dot(B_task,K)).T,M) \
                  - np.dot(M,A_task+np.dot(B_task,K)) ) , "Failed test (A+BK)'M+M(A+BK) <= -2lmbdaM"

assert np.all(C_lip == np.dot(C_task,P)) , "Failed test C_lip = C_task*P"

assert np.all( np.dot(P,A_lip) == np.dot(A_task,P) + np.dot(B_task,Q) ) \
            , "Failed Test P*A_lip = A_task*P+B*Q"

###############################################
# MPC formulation
###############################################

def perform_template_mpc(t, x_lip_init, x_task_init):
    # Prediction horizon and sampling times
    N = 50
    dt = 0.1

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

        # Simulation function
        xPx_now = (x_task[:,i] - np.dot(P,x_lip[:,i]))[np.newaxis].T
        V_now = np.dot(np.dot(xPx_now.T,M),xPx_now)  # (x_task - P*x_lip)'*M*(x_task-P*x_lip)

        
        xPx_next = (x_task[:,i+1] - np.dot(P,x_lip[:,i+1]))[np.newaxis].T
        V_next = np.dot(np.dot(xPx_next.T,M),xPx_next)  # (x_task - P*x_lip)'*M*(x_task-P*x_lip)

        # Add relaxed interface constraint
        #mp.AddConstraint((V_next[0,0]-V_now[0,0])/dt <= -lmbda*V_now[0,0])
        #mp.AddConstraint(V_now[0,0] <= 5)

        # Add interface constraint
        #A_interface = np.hstack([R, (Q-np.dot(K,P)), K, -np.eye(6)])
        #x_interface = np.hstack([u_lip[:,i],x_lip[:,i],x_task[:,i],u_task[:,i]])[np.newaxis].T
        #mp.AddLinearEqualityConstraint(A_interface, np.zeros((6,1)), x_interface)


    # Add terminal cost
    mp.AddQuadraticErrorCost(Qf_mpc,np.zeros((2,1)),x_lip[2:4,N-1])  # Penalize CoM accelrations
        
    # Solve the QP
    st = time.time()
    solver = OsqpSolver()
    #solver = GurobiSolver()
    res = solver.Solve(mp,None,None)

    print("Solve time: %ss" % (time.time()-st))
    #res = Solve(mp)
    print(res.get_solver_id().name())

    x_lip_traj = res.GetSolution(x_lip)
    u_lip_traj = res.GetSolution(u_lip)
    x_task_traj = res.GetSolution(x_task)
    u_task_traj = res.GetSolution(u_task)
    
    return x_lip_traj, u_lip_traj, x_task_traj, u_task_traj

###############################################
# Run simulation
###############################################

sim_time = step_time*zmp_ref.shape[1]
sim_time = 10.0
dt = 0.1
n_steps = int(sim_time/dt)

# For storing results
p_zmp_ref = np.zeros((n_steps,2))
p_zmp_lip = np.zeros((n_steps,2))
p_com_lip = np.zeros((n_steps,2))
p_com_task = np.zeros((n_steps,2))
output_err = np.zeros(n_steps)
sim_fcn = np.zeros(n_steps)

# Initial conditions
x_lip = np.zeros((4,1))
x_task = np.zeros((9,1)) + 0.01

# one solve version
x_lip_traj, u_lip_traj, x_task_traj, u_task_traj = perform_template_mpc(0, x_lip, x_task)

p_com_lip = x_lip_traj[0:2,:].T
p_com_task = x_task_traj[0:2,:].T
p_zmp_lip = u_lip_traj[0:2,:].T
p_zmp_ref = np.zeros(p_zmp_lip.shape)

for i in range(p_zmp_lip.shape[0]):
    t = i*0.1  # dt from MPC solver
    p_zmp_ref[i,:] = zmp_trajectory.value(t).flatten()
    
    x_lip = x_lip_traj[:,i][np.newaxis].T
    u_lip = u_lip_traj[:,i][np.newaxis].T
    x_task = x_task_traj[:,i][np.newaxis].T
    u_task = u_task_traj[:,i][np.newaxis].T

    output_err[i] = np.linalg.norm( np.dot(C_lip,x_lip)-np.dot(C_task,x_task) )
   
    x_Px = x_task - np.dot(P,x_lip)
    sim_fcn[i] = np.sqrt( np.dot( np.dot(x_Px.T,M), x_Px) )


# Series of solves version
#for i in range(n_steps):
#    t = i*dt
#
#    # Perform mpc
#    x_lip_traj, u_lip_traj, x_task_traj, u_task_traj = perform_template_mpc(t, x_lip, x_task)
#
#    # Extract relevant values
#    x_lip = x_lip_traj[:,0][np.newaxis].T
#    u_lip = u_lip_traj[:,0][np.newaxis].T
#    x_task = x_task_traj[:,0][np.newaxis].T
#    u_task = u_task_traj[:,0][np.newaxis].T
#
#    # Record plottable values
#    p_zmp_ref[i,:] = zmp_trajectory.value(t).flatten()
#    p_zmp_lip[i,:] = u_lip.flatten()
#    p_com_lip[i,:] = x_lip[0:2].flatten()
#    p_com_task[i,:] = x_task[0:2].flatten()
#
#    output_err[i] = np.linalg.norm( np.dot(C_lip,x_lip)-np.dot(C_task,x_task) )
#   
#    x_Px = x_task - np.dot(P,x_lip)
#    sim_fcn[i] = np.sqrt( np.dot( np.dot(x_Px.T,M), x_Px) )
#
#    # Simulate systems forward in time with Forward Euler
#    x_lip = x_lip + dt*(np.dot(A_lip,x_lip) + np.dot(B_lip,u_lip))
#    x_task = x_task + dt*(np.dot(A_task,x_task) + np.dot(B_task,u_task))


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

plt.legend()


# Output error and simulation function
plt.figure()
plt.plot(np.arange(0,sim_time,dt),output_err, label="Output Error")
plt.plot(np.arange(0,sim_time,dt),sim_fcn, label="Simulation Function")
plt.xlabel("Time")

plt.legend()


plt.show()

