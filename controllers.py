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

        self.mu = 0.3  # assumed friction coefficient

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
        contact_jacobians_dot_v = []

        # Right foot contacts
        for foot_index in [right_foot_index, left_foot_index]:
            for contact_point in self.get_foot_contact_points():
                J = self.tree.transformPointsJacobian(cache,             # kinematics cache
                                                      contact_point,     # point in foot frame
                                                      foot_index,        # foot frame index
                                                      world_index,       # world frame index
                                                      False)             # in terms of qd

                Jd_v = self.tree.transformPointsJacobianDotTimesV(cache,
                                                                  contact_point,
                                                                  foot_index,
                                                                  world_index)
                                                                      
                contact_jacobians.append(J)
                contact_jacobians_dot_v.append(Jd_v)

        return (contact_jacobians, contact_jacobians_dot_v)

    def solve_QP(self, cache, contact_jacobians, contact_jacobians_dot_v, xdd_com_des, qdd_des):
        """
        Solve a quadratic program which attempts to regulate the joints to the desired
        accelerations and center of mass to the desired position as follows:

            min   w_1*| Jcom*qdd + Jcomd*qd - xdd_com_des |^2 + w_2*| qdd_des - qdd |^2
            s.t.  H*qdd + C = B*tau + sum(J'*f)
                  f \in friction cones
                  J*qdd + Jd*qd == 0  for all contacts
        """
        mp = MathematicalProgram()
        num_contacts = len(contact_jacobians)

        # Set weightings for prioritized task-space control
        w1 = 1.0   # center-of-mass tracking
        w2 = 0.1   # joint tracking

        # Get dynamic quantities
        H = self.tree.massMatrix(cache)
        C = self.tree.dynamicsBiasTerm(cache, {}, True)
        B = self.tree.B

        Jcom = self.tree.centerOfMassJacobian(cache)
        Jcomd_qd = self.tree.centerOfMassJacobianDotTimesV(cache)

        # create optimization variables
        qdd = mp.NewContinuousVariables(self.nv, 'qdd')   # joint accelerations
        tau = mp.NewContinuousVariables(self.nu, 'tau')   # applied torques
       
        f_contact = {}
        for i in range(num_contacts):        # contact forces
            f_contact[i] = mp.NewContinuousVariables(3, 'f_%s'%i)

        # Cast vectors as nx1 numpy arrays to allow for matrix multiplication with np.dot()
        C = C[np.newaxis].T
        Jcomd_qd = Jcomd_qd[np.newaxis].T
        xdd_com_des = xdd_com_des[np.newaxis].T
        tau = tau[np.newaxis].T
        qdd = qdd[np.newaxis].T
        qdd_des = qdd_des[np.newaxis].T

        # Center of mass tracking cost
        x_com_err = np.dot(Jcom,qdd) + Jcomd_qd - xdd_com_des
        x_com_cost = np.dot(x_com_err.T,x_com_err)    # this is a 1x1 np.ndarray and we need a pydrake.symbolic.Expression
        x_com_cost = x_com_cost[0,0]                  # to use as a cost, so we simply extract the first element.

        mp.AddQuadraticCost(w1*x_com_cost)
       
        # Joint tracking cost
        q_err = qdd - qdd_des
        q_cost = np.dot(q_err.T,q_err)
        q_cost = q_cost[0,0]

        mp.AddQuadraticCost(w2*q_cost)
        
        # Dynamic constraints 
        f_ext = sum([np.dot(contact_jacobians[i].T, f_contact[i][np.newaxis].T) for i in range(num_contacts)])
        lhs = np.dot(H,qdd) + C
        rhs = np.dot(B,tau) + f_ext

        for i in range(self.nv):
            mp.AddLinearConstraint(lhs[i,0] == rhs[i,0])

        # Friction cone (really pyramid) constraints 
        for j in range(num_contacts):
            mp.AddConstraint(f_contact[j][0] + f_contact[j][1] <= self.mu*f_contact[j][2])
            mp.AddConstraint(-f_contact[j][0] + f_contact[j][1] <= self.mu*f_contact[j][2])
            mp.AddConstraint(f_contact[j][0] - f_contact[j][1] <= self.mu*f_contact[j][2])
            mp.AddConstraint(-f_contact[j][0] - f_contact[j][1] <= self.mu*f_contact[j][2])

        # Contact acceleration constraints
        for j in range(num_contacts):
            J = contact_jacobians[j]
            Jd_qd = contact_jacobians_dot_v[j][np.newaxis].T

            contact_accel = np.dot(J,qdd) + Jd_qd
            for i in range(3):
                mp.AddConstraint(contact_accel[i,0] == 0)
       
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

        # Run kinematics, which will allow us to calculate the dynamics
        cache = self.tree.doKinematics(q,qd)

        # Compute contact jacobians
        contact_jacobians, contact_jacobians_dot_v = self.get_contact_jacobians(cache)

        # Compute desired joint acclerations
        q_nom = self.nominal_state[:self.np]
        qd_nom = self.nominal_state[self.np:]

        qdd_des = 100*(q_nom-q) + 1*(qd_nom-qd)

        # Compute desired center of mass acceleration
        x_com = self.tree.centerOfMass(cache)
        xd_com = np.dot(self.tree.centerOfMassJacobian(cache), qd)
        x_com_nom = np.asarray([ 0.0, 0.0, 1.04 ])
        xd_com_nom = np.asarray([ 0.0, 0.0, 0.0 ])

        print(x_com)

        xdd_com_des = 100*(x_com_nom-x_com) + 10*(xd_com_nom-xd_com)

        # Solve QP to get desired torques
        tau_qp = self.solve_QP(cache, contact_jacobians, contact_jacobians_dot_v, xdd_com_des, qdd_des)

        output[:] = tau_qp
        

