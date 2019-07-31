##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import numpy as np
from pydrake.all import VectorSystem
from helpers import RPYValkyrieFixedPointTorque, RPYValkyrieFixedPointState

class ValkyrieController(VectorSystem):
    """
    PD Controller that attempts to regulate the robot to a fixed initial position
    """
    def __init__(self, input_size=73, output_size=30):
        VectorSystem.__init__(self, 
                              input_size,   # input size [q,qd]
                              output_size)   # output size [tau]

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
        self.nominal_state = RPYValkyrieFixedPointState()

    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        #q = state[:36]
        #qd = state[36:]

        #q_nom = self.nominal_state[:36]
        #qd_nom = self.nominal_state[36:]

        #tau_ff = RPYValkyrieFixedPointTorque()
        #tau_pd = np.matmul(self.Kp,(q_nom-q)) + np.matmul(self.Kd, (qd_nom-q))

        #output[:] = tau_ff + tau_pd[6:]
        output[:] = 0

