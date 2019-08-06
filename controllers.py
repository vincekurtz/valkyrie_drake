##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import numpy as np
import time
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
        self.nu = tree.get_num_actuators()

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

    def get_contact_jacobians(self, cache):
        """
        Return a list of contact jacobians for both feet, assuming double support and that
        each foot has contacts in the corners, as defined by self.get_foot_contact_points()
        """
        right_foot_index = self.tree.FindBody('rightFoot').get_body_index()
        left_foot_index = self.tree.FindBody('leftFoot').get_body_index()
        world_index = self.tree.world().get_body_index()

        contact_jacobians = []

        # Right foot contacts
        for contact_point in self.get_foot_contact_points():
            contact_jacobian = self.tree.transformPointsJacobian(cache,             # kinematics cache
                                                                 contact_point,     # point in foot frame
                                                                 right_foot_index,  # foot frame index
                                                                 world_index,       # world frame index
                                                                 False)             # in terms of qd
            contact_jacobians.append(contact_jacobian)

        # Left foot contacts
        for contact_point in self.get_foot_contact_points():
            contact_jacobian = self.tree.transformPointsJacobian(cache,
                                                                 contact_point,
                                                                 left_foot_index,
                                                                 world_index,
                                                                 False)
            contact_jacobians.append(contact_jacobian)

        return contact_jacobians

    def solve_QP(self, H, C, B, contact_jacobians, qdd_des):
        """
        Solve a quadratic program which attempts to regulate the joints to the desired
        accelerations, qdd_des, as follows:

            min  \| qdd - qdd_des \|^2
            s.t.  H*qdd + C = B*tau + sum(J'*f)
                  f \in friction cones
        """
        mp = MathematicalProgram()

        # create optimization variables
        qdd = mp.NewContinuousVariables(self.nv, 'qdd')   # joint accelerations
        tau = mp.NewContinuousVariables(self.nu, 'tau')   # applied torques
       
        f_contact = {}
        for i in range(len(contact_jacobians)):        # contact forces
            f_contact[i] = mp.NewContinuousVariables(3, 'f_%s'%i)

        # Define the cost function
        qdd_des = qdd_des[np.newaxis].T    # cast as vectors for numpy
        qdd = qdd[np.newaxis].T
        I = np.matrix(np.eye(self.nv))

        cost = (qdd_des.T-qdd.T)*I*(qdd_des-qdd)    # this is a 1x1 np.matrix and we need a pydrake.symbolic.Expression
        cost = cost[0,0]                            # to use as a cost, so we simply extract the first element.

        mp.AddQuadraticCost(cost)

        # Dynamic constraints 
        C = C[np.newaxis].T         # cast as vectors for numpy multiplication
        tau = tau[np.newaxis].T

        f_ext = sum([np.dot(contact_jacobians[i].T, f_contact[i][np.newaxis].T) for i in range(len(contact_jacobians))])
        lhs = np.dot(H,qdd) + C
        rhs = np.dot(B,tau) + f_ext

        for i in range(self.nv):
            mp.AddLinearConstraint(lhs[i,0] == rhs[i,0])

        # Friction cone constraints 
        for j in range(len(contact_jacobians)):
            mp.AddLinearConstraint(f_contact[j][2] >= 0)   # positive z contact force
       
        # Solve the QP
        result = Solve(mp)

        assert result.is_success(), "Whole-body QP Solver Failed!"

        return result.GetSolution(tau)


    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        q = state[:self.np]
        qd = state[self.np:]

        q_nom = self.nominal_state[:self.np]
        qd_nom = self.nominal_state[self.np:]

        # Get equations of motion
        #
        #  H(q)*qdd + C(q,qd) = B*tau + f_ext
        #
        cache = self.tree.doKinematics(q,qd)
        H = self.tree.massMatrix(cache)
        C = self.tree.dynamicsBiasTerm(cache, {}, True)
        B = self.tree.B

        # Compute contact jacobians
        contact_jacobians = self.get_contact_jacobians(cache)

        # Get centroidal momentum quantities
        A = self.tree.centroidalMomentumMatrix(cache)
        Ad_qd = self.tree.centroidalMomentumMatrixDotTimesV(cache)

        # Solve QP to get desired torques
        qdd_des = 100*(q_nom-q) + 1*(qd_nom-qd)
        tau_qp = self.solve_QP(H, C, B, contact_jacobians, qdd_des)

        output[:] = tau_qp
        

