#!/usr/bin/env python2

##
#
# Finite state machines for walking and standing gaits. Each FSM specifies target CoM 
# positions, stance phases, and target foot positions as functions of time. 
#
##

import numpy as np
from copy import copy
from pydrake.all import PiecewisePolynomial

class StandingFSM(object):
    """
    A finite state machine describing simply standing in double support at the given position.
    """
    def __init__(self):
        self.x_com_init = np.asarray([[0.0], [0.0], [0.967]])
        self.x_right_init = np.asarray([[-0.071], [-0.138], [0.099]])
        self.x_left_init = np.asarray([[-0.071], [0.138], [0.099]])

    def SupportPhase(self, time):
        """
        Return the current support phase, "double", "left", or "right".
        """
        return "double"

    def ComTrajectory(self, time):
        """
        Return a desired center of mass position and velocity for the given timestep
        """
        x_com = self.x_com_init
        xd_com = np.array([[0.0],[0.0],[0.0]])
        return (x_com, xd_com)

    def RightFootTrajectory(self, time):
        """
        Specify a desired position and velocity for the right foot
        """
        x_right = self.x_right_init
        xd_right = np.array([[0.0],[0.0],[0.0]])
        return (x_right, xd_right)
    
    def LeftFootTrajectory(self, time):
        """
        Specify a desired position and velocity for the left foot
        """
        x_left = self.x_left_init
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
        self.x_right_init = np.asarray([[-0.071], [-0.138], [0.099]])
        self.x_left_init = np.asarray([[-0.071], [0.138], [0.099]])

        self.foot_width = 0.16  # for visualization purposes only
        self.foot_length = 0.27

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
                x_right = copy(self.right_foot_placements[-1])
                x_right[0,0] += l
                self.right_foot_placements.append(x_right)
            else:
                # Step with the left foot
                x_left = copy(self.left_foot_placements[-1])
                x_left[0,0] += l
                self.left_foot_placements.append(x_left)

        # Generate ZMP trajectory as function of time
        self._generate_zmp_trajectory()

        # Generate foot trajectories as functions of time.
        self._generate_foot_trajectories()

    def create_plottable_foot_polygon(self, x_foot, alpha=1.0):
        """
        Return a Poly3DCollection object representing a foot
        at the specified position.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        x = [x_foot[0,0] + self.foot_length/2,
             x_foot[0,0] + self.foot_length/2,
             x_foot[0,0] - self.foot_length/2,
             x_foot[0,0] - self.foot_length/2]
        y = [x_foot[1,0] + self.foot_width/2,
             x_foot[1,0] - self.foot_width/2,
             x_foot[1,0] - self.foot_width/2,
             x_foot[1,0] + self.foot_width/2]
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

        timesteps = np.arange(0,self.total_time,0.1)

        rf_poly = None
        lf_poly = None
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

            plt.draw()
            plt.pause(0.01)

        plt.show()
    
    def _generate_zmp_trajectory(self):
        """
        Compute a desired ZMP trajectory as a piecewise linear function. Once this
        method has been called, the ZMP at a given timestep t can be accessed as

            [x_zmp, y_zmp] = self.zmp_trajectory.value(t).

        """
        # Specify break points for piecwise linear interpolation
        break_times = np.asarray([[i*self.step_time] for i in range(self.n_phases+1)])

        initial_x = np.mean([self.right_foot_placements[0][0], self.left_foot_placements[0][0]])
        initial_y = np.mean([self.right_foot_placements[0][1], self.left_foot_placements[0][1]])

        # specify the desired ZMP at the break times
        zmp_ref = np.asarray([[initial_x, initial_y]])

        # Shift ZMP to under the left foot initially
        zmp_ref = np.vstack([zmp_ref,self.left_foot_placements[0][0:2,0]])
        zmp_ref = np.vstack([zmp_ref,self.left_foot_placements[0][0:2,0]])

        rf_idx = 1
        lf_idx = 1
        for i in range(self.n_steps-1):
            if i % 2 == 0:
                # Right foot moved: shift ZMP under the right foot now
                foot_center = self.right_foot_placements[rf_idx]
                rf_idx += 1
            else:
                # Left foot moved: shift ZMP under the left foot now
                foot_center = self.left_foot_placements[lf_idx]
                lf_idx += 1

            zmp_ref = np.vstack([zmp_ref,foot_center[0:2,0]])
            zmp_ref = np.vstack([zmp_ref,foot_center[0:2,0]])

        # Final shift of ZMP to under support area in double support
        last_x = zmp_ref[-1,0]
        zmp_ref = np.vstack([zmp_ref, np.array([last_x, 0])])

        # ZMP reference knot points must be formatted such that each column is a knot point
        zmp_ref = zmp_ref.T

        self.zmp_trajectory = PiecewisePolynomial.FirstOrderHold(break_times,zmp_ref)

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

        lf_idx = 0
        rf_idx = 0
        for i in range(len(break_times)):
            t = break_times[i,0]
            phase_time = (t/self.step_time) % 4
            if 1 < phase_time and phase_time < 2:
                # Right foot is in the middle of its swing phase
                rf_ref = copy(self.right_foot_placements[rf_idx])
                rf_ref[0] += self.step_length/2  # move the foot forward to the midpoint of the stride
                rf_ref[2] += self.step_height    # and up the designated hight

                # Next setpoint will be at the next foot placement
                rf_idx += 1

                # left foot remains at the same place
                lf_ref = self.left_foot_placements[lf_idx]

            elif 3 < phase_time and phase_time < 4:
                # Left foot is in the middle of its swing phase
                lf_ref = copy(self.left_foot_placements[lf_idx])
                lf_ref[0] += self.step_length/2  # move the foot forward to the midpoint of the stride
                lf_ref[2] += self.step_height    # and up the designated hight

                # Next setpoint will be at the next foot placement
                lf_idx += 1

                # right foot remains at the same place
                rf_ref = self.right_foot_placements[rf_idx]
            else:
                # We're in double support: both feet stay at the same position
                lf_ref = self.left_foot_placements[lf_idx]
                rf_ref = self.right_foot_placements[rf_idx]

            lf_knots[:,i] = lf_ref[:,0]
            rf_knots[:,i] = rf_ref[:,0]

        # Perform piecewise linear interpolation
        self.left_foot_trajectory = PiecewisePolynomial.FirstOrderHold(break_times,lf_knots)
        self.right_foot_trajectory = PiecewisePolynomial.FirstOrderHold(break_times,rf_knots)

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

    def ComTrajectory(self, time):
        """
        Return a desired center of mass position and velocity for the given timestep
        """
        # TODO: specify trajectory that tracks the ZMP trajectory

        x_com = self.x_com_init
        xd_com = np.array([[0.0],[0.0],[0.0]])
        return (x_com, xd_com)

    def RightFootTrajectory(self, time):
        """
        Specify a desired position and velocity for the right foot
        """
        x_right = self.x_right_init
        xd_right = np.array([[0.0],[0.0],[0.0]])
        return (x_right, xd_right)
    
    def LeftFootTrajectory(self, time):
        """
        Specify a desired position and velocity for the left foot
        """
        x_left = self.x_left_init
        xd_left = np.array([[0.0],[0.0],[0.0]])
        return (x_left, xd_left)


if __name__=="__main__":
    # test generating a trajectory
    n_steps = 4
    step_length = 0.5
    step_height = 0.1
    step_time = 1.0

    fsm = WalkingFSM(n_steps, step_length, step_height, step_time)

    # Debugging: plot the ZMP trajectory
    fsm.plot_walking_pattern()
