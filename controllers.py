##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import numpy as np
from pydrake.all import *
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

    def get_foot_contact_points(self):
        """
        Return a tuple of points in the foot frame that represent contact locations. 
        (this is a rough guess based on self.tree.getTerrainContactPoints)
        """
        corner_contacts = (
                            [-0.069, 0.08, 0.0],
                            [-0.069,-0.08, 0.0],
                            [ 0.201,-0.08, 0.0],
                            [ 0.201, 0.08, 0.0]
                          )

        return corner_contacts
        

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

        right_foot = self.tree.FindBody('rightFoot')
        left_foot = self.tree.FindBody('leftFoot')

        # Get the position of each foot
        world_body_index = self.tree.world().get_body_index()
        left_foot_index = left_foot.get_body_index()
        left_foot_in_world_frame = self.tree.transformPoints(cache,      # kinematics cache
                                                            [0,0,0],     # point relative the foot frame
                                                            left_foot_index,  # foot frame index
                                                            world_body_index) # world frame index

        left_foot_jacobian = self.tree.transformPointsJacobian(cache,
                                                                [0,0,0],
                                                                left_foot_index,
                                                                world_body_index,
                                                                False)


        collisions = self.tree.ComputeMaximumDepthCollisionPoints(cache)

        terrain_contacts = self.tree.getTerrainContactPoints(right_foot)

        for i in range(terrain_contacts.shape[1]):
            print(repr(terrain_contacts[:,i]))
        print("")

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
        

