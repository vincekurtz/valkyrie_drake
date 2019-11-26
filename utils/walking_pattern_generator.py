#!/usr/bin/env python2

##
#
# Finite state machines for walking and standing gaits. Each FSM specifies target CoM 
# positions, stance phases, and target foot positions as functions of time. 
#
##

import numpy as np
from copy import copy as pycopy
from pydrake.all import *

class StandingFSM(object):
    """
    A finite state machine describing simply standing in double support at the given position.
    """
    def __init__(self):
        self.x_com_init = np.asarray([[0.0], [0.0], [0.967]])
        self.x_right_init = np.asarray([[-0.065], [-0.138], [0.1]])
        self.x_left_init = np.asarray([[-0.065], [0.138], [0.1]])

        # define ZMP trajectory
        self.zmp_trajectory = type('', (), {})()  # define a new class in one line so we can imitate
                                                  # the results using a polynomial interpolation, in 
                                                  # which case we access the zmp trajectory at time t 
                                                  # via zmp_trajectory.value(t)
        self.zmp_trajectory.value = lambda t : np.zeros((2,1))

    def SupportPhase(self, time):
        """
        Return the current support phase, "double", "left", or "right".
        """
        return "double"

    def ComTrajectory(self, time):
        """
        Return a desired center of mass position and velocity for the given timestep
        """
        x_com = pycopy(self.x_com_init)
        xd_com = np.array([[0.0],[0.0],[0.0]])
        xdd_com = np.array([[0.0],[0.0],[0.0]])
        return (x_com, xd_com, xdd_com)

    def RightFootTrajectory(self, time):
        """
        Specify a desired position and velocity for the right foot
        """
        x_right = pycopy(self.x_right_init)
        xd_right = np.array([[0.0],[0.0],[0.0]])
        return (pycopy(x_right), pycopy(xd_right))
    
    def LeftFootTrajectory(self, time):
        """
        Specify a desired position and velocity for the left foot
        """
        x_left = pycopy(self.x_left_init)
        xd_left = np.array([[0.0],[0.0],[0.0]])
        return (x_left, xd_left)

class WalkingFSM(object):
    """
    A finite state machine describing a simple walking motion. Specifies support phase,
    foot positions, and center of mass position as functions of time. 
    """
    def __init__(self, n_steps, step_length, step_height, step_time):

        assert n_steps >= 2, "Must specify at least two steps"

        self.n_steps = n_steps          # number of steps to take
        self.step_length = step_length  # how far forward each step goes
        self.step_height = step_height  # maximum height of the swing foot
        self.step_time = step_time      # how long each swing phase lasts

        self.n_phases = 2*n_steps + 1     # how many different phases (double support, left support, 
                                          # right support, etc. there are throughout the whole motion.
        self.total_time = self.step_time*self.n_phases  

        # initial CoM and foot positions
        self.x_com_init = np.asarray([[0.0], [0.0], [0.967]])
        self.xd_com_init = np.asarray([[0.0], [0.0], [0.0]])

        self.fc_offset = -0.065   # The foot frame is this far from the foot's center

        self.x_right_init = np.asarray([[self.fc_offset], [-0.138], [0.1]])
        self.x_left_init = np.asarray([[self.fc_offset], [0.138], [0.1]])

        self.foot_w1 = 0.08  # width left of foot frame
        self.foot_w2 = 0.08  # width right of foot frame
        self.foot_l1 = 0.2   # length in front of foot
        self.foot_l2 = 0.07  # length behind foot

        # LIP parameters
        self.h = self.x_com_init[2]  
        self.g = 9.81

        # Generate foot ground contact placements for both feet
        self.right_foot_placements = [self.x_right_init]
        self.left_foot_placements = [self.x_left_init]

        for step in range(self.n_steps):
            if (step == 0) or (step == self.n_steps-1):
                # This is the first or last step, so shorten the stride
                l = self.step_length/2
            else:
                l = self.step_length
            if step % 2 == 0:
                # Step with the right foot
                x_right = pycopy(self.right_foot_placements[-1])
                x_right[0,0] += l
                self.right_foot_placements.append(x_right)
            else:
                # Step with the left foot
                x_left = pycopy(self.left_foot_placements[-1])
                x_left[0,0] += l
                self.left_foot_placements.append(x_left)

        # Generate ZMP trajectory as function of time
        self._generate_zmp_trajectory()

        # Generate CoM trajectory as function of time using the LIP
        self._generate_com_trajectory()

        # Generate foot trajectories as functions of time.
        self._generate_foot_trajectories()


    def create_plottable_foot_polygon(self, x_foot, alpha=1.0):
        """
        Return a Poly3DCollection object representing a foot
        at the specified position.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        x = [x_foot[0,0] + self.foot_l1,
             x_foot[0,0] + self.foot_l1,
             x_foot[0,0] - self.foot_l2,
             x_foot[0,0] - self.foot_l2]
        y = [x_foot[1,0] + self.foot_w1,
             x_foot[1,0] - self.foot_w2,
             x_foot[1,0] - self.foot_w2,
             x_foot[1,0] + self.foot_w1]
        z = [x_foot[2,0] for i in range(4)]

        verts = [list(zip(x, y, z))]
        return Poly3DCollection(verts, alpha=alpha)

    def plot_walking_pattern(self):
        """
        Make a quick animation/plot of the walking pattern.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.view_init(30,60)

        timesteps = np.arange(0,self.total_time,0.05)

        rf_poly = None
        lf_poly = None
        x_com = self.x_com_init
        xd_com = self.xd_com_init
        for t in timesteps:

            # Remove previous foot polygons.
            if rf_poly is not None:
                rf_poly.remove()
            if lf_poly is not None:
                lf_poly.remove()

            # Plot foot positions
            p_rf = self.right_foot_trajectory.value(t)
            p_lf = self.left_foot_trajectory.value(t)
            rf_poly = self.create_plottable_foot_polygon(p_rf, alpha=0.5)
            lf_poly = self.create_plottable_foot_polygon(p_lf, alpha=0.5)

            ax.add_collection3d(rf_poly)
            ax.add_collection3d(lf_poly)

            # Plot ZMP position
            p_zmp = self.zmp_trajectory.value(t)
            ax.scatter(p_zmp[0],p_zmp[1], 0.099)

            # Plot CoM position
            x_com = self.com_trajectory.value(t)
            ax.scatter(x_com[0], x_com[1], x_com[2])

            plt.draw()
            plt.pause(0.001)

        plt.show()
    
    def _generate_zmp_trajectory(self):
        """
        Compute a desired ZMP trajectory as a piecewise linear function. Once this
        method has been called, the ZMP at a given timestep t can be accessed as
            [x_zmp, y_zmp] = self.zmp_trajectory.value(t).
        """
        # Specify break points for piecwise linear interpolation
        break_times = np.asarray([[i*self.step_time] for i in range(self.n_phases+1)])

        # Initial x position accounts for fc_offset, the difference between center of foot and origin of
        # foot frame in the Valkyrie model.
        initial_x = np.mean([self.right_foot_placements[0][0], self.left_foot_placements[0][0]]) - self.fc_offset
        initial_y = np.mean([self.right_foot_placements[0][1], self.left_foot_placements[0][1]])

        # specify the desired ZMP at the break times
        zmp_ref = np.asarray([[initial_x, initial_y]])

        # Shift ZMP to under the left foot initially
        zmp_ref = np.vstack([zmp_ref,self.left_foot_placements[0][0:2,0] - [self.fc_offset, 0]])

        # Wait under left foot
        zmp_ref = np.vstack([zmp_ref,self.left_foot_placements[0][0:2,0] - [self.fc_offset, 0]])

        rf_idx = 1
        lf_idx = 1
        for i in range(self.n_steps-1):
            if i % 2 == 0:
                # Right foot moved: shift ZMP under the right foot now
                foot_center = self.right_foot_placements[rf_idx] - np.asarray([[self.fc_offset, 0, 0]]).T
                rf_idx += 1
            else:
                # Left foot moved: shift ZMP under the left foot now
                foot_center = self.left_foot_placements[lf_idx] - np.asarray([[self.fc_offset, 0, 0]]).T
                lf_idx += 1

            zmp_ref = np.vstack([zmp_ref,foot_center[0:2,0]])
            zmp_ref = np.vstack([zmp_ref,foot_center[0:2,0]])

        # Final shift of ZMP to under support area in double support
        last_x = zmp_ref[-1,0]
        zmp_ref = np.vstack([zmp_ref, np.array([last_x, 0])])

        # ZMP reference knot points must be formatted such that each column is a knot point
        zmp_ref = zmp_ref.T

        self.zmp_trajectory = PiecewisePolynomial.FirstOrderHold(break_times,zmp_ref)

    def _generate_com_trajectory(self):
        """
        Compute a desired CoM trajectory that tracks the desired ZMP trajectory and
        parameterize as a piecwise polynomial. Once this method has been called, the
        center of mass position, velocity, and accleration at a given timestep t can 
        be accessed as

            x_com = self.com_trajectory.value(t)
            xd_com = self.com_trajectory.derivative(1).value(t)
            xdd_com = self.com_trajectory.derivative(2).value(t).
        """
        dt = 0.1
        n_steps = int(self.total_time/dt)

        # Solve an MPC problem over the whole trajectory
        x, u, y = self.ZMPComMPC(0.0, self.x_com_init, self.xd_com_init, dt=dt, n_steps=n_steps)

        # Extract center of mass trajectory, adding z axis back in
        x_com_trajectory = np.vstack([x[0:2,:],self.h*np.ones((1,n_steps))])
        xd_com_trajectory = np.vstack([x[2:4,:],np.zeros((1,n_steps))])
        xdd_com_trajectory = np.vstack([u,np.zeros((1,n_steps))])

        # Specify break points for polynomial interpolation
        break_times = np.asarray([[i*dt] for i in range(n_steps)])

        # Perform polynomial interpolation
        self.com_trajectory = PiecewisePolynomial.Cubic(break_times,x_com_trajectory,xd_com_trajectory)

    def _generate_foot_trajectories(self):
        """
        Generate a desired (x,y,z) trajectory for each foot as a piecewise polynomial.
        Once this method has been called, the foot positions at a given time t can be
        accessed like
            left_foot_position = self.left_foot_trajectory.value(t).
        """
        # Specify break points for piecewise linear interpolation
        break_times = np.asarray([[0.5*i*self.step_time] for i in range(2*self.n_phases+1)])

        # Knot points are positions of the right and left feet at the break times
        lf_knots = np.zeros((3,len(break_times)))
        rf_knots = np.zeros((3,len(break_times)))
        
        lf_dot_knots = np.zeros((3,len(break_times)))
        rf_dot_knots = np.zeros((3,len(break_times)))

        lf_idx = 0
        rf_idx = 0
        for i in range(len(break_times)):
            t = break_times[i,0]
            phase_time = (t/self.step_time) % 4
            if 1 < phase_time and phase_time < 2:
                # Right foot is in the middle of its swing phase
                rf_ref = pycopy(self.right_foot_placements[rf_idx])
                rf_ref[0] += self.step_length/2  # move the foot forward to the midpoint of the stride
                rf_ref[2] += self.step_height    # and up the designated hight

                # Slight forward velocity of the swing foot
                rf_dot_knots[0,i] = self.step_length/self.step_time

                # Next setpoint will be at the next foot placement
                rf_idx += 1

                # left foot remains at the same place
                lf_ref = self.left_foot_placements[lf_idx]

            elif 3 < phase_time and phase_time < 4:
                # Left foot is in the middle of its swing phase
                lf_ref = pycopy(self.left_foot_placements[lf_idx])
                lf_ref[0] += self.step_length/2  # move the foot forward to the midpoint of the stride
                lf_ref[2] += self.step_height    # and up the designated hight

                # Next setpoint will be at the next foot placement
                lf_idx += 1
                
                # Slight forward velocity of the swing foot
                lf_dot_knots[0,i] = self.step_length/self.step_time

                # right foot remains at the same place
                rf_ref = self.right_foot_placements[rf_idx]
            else:
                # We're in double support: both feet stay at the same position
                lf_ref = self.left_foot_placements[lf_idx]
                rf_ref = self.right_foot_placements[rf_idx]

            lf_knots[:,i] = lf_ref[:,0]
            rf_knots[:,i] = rf_ref[:,0]

        # Perform piecewise linear interpolation
        #self.left_foot_trajectory = PiecewisePolynomial.FirstOrderHold(break_times,lf_knots)
        #self.right_foot_trajectory = PiecewisePolynomial.FirstOrderHold(break_times,rf_knots)
        self.left_foot_trajectory = PiecewisePolynomial.Cubic(break_times,lf_knots,lf_dot_knots)
        self.right_foot_trajectory = PiecewisePolynomial.Cubic(break_times,rf_knots,rf_dot_knots)

    def SupportPhase(self, time):
        """
        Return the current support phase, "double", "left", or "right".
        """

        if time >= self.total_time:
            return "double"
        if time/self.step_time % 4 <= 1:
            return "double"
        elif time/self.step_time % 4 <= 2:
            return "left"
        elif time/self.step_time % 4 <= 3:
            return "double"
        elif time/self.step_time % 4 <= 4:
            return "right" 

    def ZMPComMPC(self, time, x_com, xd_com, dt=0.1, n_steps=10):
        """
        Solve an MPC problem to determine CoM position, velocity, and acceleration
        that will track the desired ZMP trajectory at the given timestep.
        """
        # LIP Dynamics in discrete time
	A = np.eye(4)
	A[0:2,2:4] = dt*np.eye(2)

	B = np.vstack([np.eye(2),np.eye(2)])
        B[0:2,0:2] *= 0.5*(dt**2)
        B[2:4,0:2] *= dt

	C = np.zeros((2,4))
	C[0:2,0:2] = np.eye(2)

	D = -self.h/self.g*np.eye(2)

        # Set up the Trajectory optimization problem
        mp = MathematicalProgram()
        x = mp.NewContinuousVariables(4,n_steps,'x')  # CoM position and velocity
        y = mp.NewContinuousVariables(2,n_steps,'y')  # zero moment point
        u = mp.NewContinuousVariables(2,n_steps,'u')  # LIP center-of-pressure

        # Initial constraint
        mp.AddLinearEqualityConstraint(np.eye(2), x_com[0:2], x[0:2,0])
        mp.AddLinearEqualityConstraint(np.eye(2), xd_com[0:2], x[2:4,0])

        # Cost parameters
        Q = 20*np.eye(2)
        Qf = 100*np.eye(2)
        R = np.eye(2)

        for i in range(n_steps-1):
            # Dynamic Constraint
            A_bar = np.hstack((A,B,-np.eye(4)))
            x_bar = np.hstack((x[:,i],u[:,i],x[:,i+1]))[np.newaxis].T
            mp.AddLinearEqualityConstraint(A_bar,np.zeros((4,1)),x_bar)

            # Output Constraint
            A_bar = np.hstack((C,D,-np.eye(2)))
            x_bar = np.hstack((x[:,i],u[:,i],y[:,i]))[np.newaxis].T
            mp.AddLinearEqualityConstraint(A_bar,np.zeros((2,1)),x_bar)

            # Cost
            y_des = self.zmp_trajectory.value(time+dt*i)
            mp.AddQuadraticErrorCost(Q, y_des, y[:,i])

            mp.AddQuadraticCost(R, np.zeros((2,1)), u[:,i])

        # Terminal cost
        y_des = self.zmp_trajectory.value(time+dt*n_steps)
        mp.AddQuadraticErrorCost(Qf,y_des, y[:,n_steps-1])
        mp.AddQuadraticCost(R, np.zeros((2,1)), u[:,n_steps-1])

        # Terminal constraint: CoM should be directly above the ZMP when we stop
        mp.AddLinearEqualityConstraint( y[0,n_steps-1] == x[0,n_steps-1] )
        mp.AddLinearEqualityConstraint( y[1,n_steps-1] == x[1,n_steps-1] )

        solver = OsqpSolver()
        res = solver.Solve(mp, initial_guess=None, solver_options=None)

        x = res.GetSolution(x)
        u = res.GetSolution(u)
        y = res.GetSolution(y)

        return (x,u,y)

    def ComTrajectory(self, time):
        """
        Return the desired position, velocity, and acceleration of the center of 
        mass at the given time.
        """
        x_com = self.com_trajectory.value(time)
        xd_com = self.com_trajectory.derivative(1).value(time)
        xdd_com = self.com_trajectory.derivative(2).value(time)

        return (x_com, xd_com, xdd_com)

    def RightFootTrajectory(self, time):
        """
        Return a desired position and velocity for the right foot
        """
        x_right = self.right_foot_trajectory.value(time)
        xd_right = self.right_foot_trajectory.derivative(1).value(time)
        
        return (x_right, xd_right)
    
    def LeftFootTrajectory(self, time):
        """
        Return a desired position and velocity for the left foot
        """
        x_left = self.left_foot_trajectory.value(time)
        xd_left = self.left_foot_trajectory.derivative(1).value(time)

        return (x_left, xd_left)


if __name__=="__main__":
    # test generating a trajectory
    n_steps = 4
    step_length = 0.5
    step_height = 0.1
    step_time = 0.5

    fsm = WalkingFSM(n_steps, step_length, step_height, step_time)

    fsm.plot_walking_pattern()

