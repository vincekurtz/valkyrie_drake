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
    def __init__(self, tree):
        VectorSystem.__init__(self, 
                              tree.get_num_positions() + tree.get_num_velocities(),   # input size [q,qd]
                              tree.get_num_actuators())   # output size [tau]

        self.tree = tree
        self.np = tree.get_num_positions()
        self.nv = tree.get_num_velocities()

        # Nominal state
        self.nominal_state = RPYValkyrieFixedPointState()

    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        q = state[:self.np]
        qd = state[self.np:]

        # Get equations of motion
        cache = self.tree.doKinematics(q,qd)
        H = self.tree.massMatrix(cache)
        tauG = -self.tree.dynamicsBiasTerm(cache, {}, False)
        Cv = self.tree.dynamicsBiasTerm(cache, {}, True) + tauG
        B = self.tree.B

        # Get centroidal momentum quantities
        A = self.tree.centroidalMomentumMatrix(cache)
        Ad_qd = self.tree.centroidalMomentumMatrixDotTimesV(cache)

        # Try to regulate to the nominal position
        q_nom = self.nominal_state[:self.np]
        qd_nom = self.nominal_state[self.np:]

        Kp = 1000
        Kd = 1
        tau_ff = RPYValkyrieFixedPointTorque()
        tau = tau_ff + Kp*(q_nom - q) + Kd*(qd_nom-qd)  

        u = np.matmul(B.T,tau)  # map desired generalized forces to control inputs

        output[:] = u
        

