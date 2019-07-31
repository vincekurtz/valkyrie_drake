##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import numpy as np
from pydrake.all import *
from helpers import *

class ValkyrieController(VectorSystem):
    """
    PD Controller that attempts to regulate the robot to a fixed initial position
    """
    def __init__(self, robot):


        VectorSystem.__init__(self, 
                              robot.num_positions()+robot.num_velocities(),   # input is [q,v]
                              robot.num_actuators())                          # output is [tau]
        
        self.robot = robot

        # PD parameters copied from drake/examples/valkyrie/valkyrie_pd_ff_controller.cc
        Kp_vec = np.asarray([0, 0, 0, 0, 0, 0])                       # base
        Kp_vec = np.hstack((Kp_vec,[100, 300, 300]))                 # spine
	Kp_vec = np.hstack((Kp_vec,[10]))                            # neck
	Kp_vec = np.hstack((Kp_vec,[10, 10, 10]))                    # r shoulder
	Kp_vec = np.hstack((Kp_vec,[1, 1, 0.1, 0.1]))                # r arm
	Kp_vec = np.hstack((Kp_vec,[10, 10, 10]))                    # l shoulder
	Kp_vec = np.hstack((Kp_vec,[1, 1, 0.1, 0.1]))                # l arm
	Kp_vec = np.hstack((Kp_vec,[100, 100, 300, 300, 300, 100]))  # r leg
	Kp_vec = np.hstack((Kp_vec,[100, 100, 300, 300, 300, 100]))  # l leg

        self.Kp = np.diag(Kp_vec)

        Kd_vec = np.asarray([0, 0, 0, 0, 0, 0])                       # base
        Kd_vec = np.hstack((Kd_vec,[10, 10, 10]))                    # spine
	Kd_vec = np.hstack((Kd_vec,[3]))                             # neck
	Kd_vec = np.hstack((Kd_vec,[3, 3, 3]))                       # r shoulder
	Kd_vec = np.hstack((Kd_vec,[0.1, 0.1, 0.01, 0.01]))          # r arm
	Kd_vec = np.hstack((Kd_vec,[3, 3, 3]))                       # l shoulder
	Kd_vec = np.hstack((Kd_vec,[0.1, 0.1, 0.01, 0.01]))          # l arm
	Kd_vec = np.hstack((Kd_vec,[10, 10, 10, 10, 10, 10]))        # r leg
	Kd_vec = np.hstack((Kd_vec,[10, 10, 10, 10, 10, 10]))        # l leg

        self.Kd = np.diag(Kd_vec)

        # Nominal state
        self.nominal_state = ValkyrieFixedPointState()

    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        q = state[0:self.robot.num_positions()]
        v = state[self.robot.num_positions():]

        M, Cv, tauG, B, tauExt = ManipulatorDynamics(self.robot, q, v)

        print(B)

        output[:] = 0
