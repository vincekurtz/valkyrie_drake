##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import numpy as np
import time
from pydrake.all import *
from helpers import *

class ZeroInputController(VectorSystem):
    """
    The most simple possible controller, one that simply outputs zeros as the control.
    """
    def __init__(self, tree):
        VectorSystem.__init__(self, 
                              tree.get_num_positions() + tree.get_num_velocities(),   # input size [q,qd]
                              tree.get_num_actuators())   # output size [tau]
    def DoCalcVectorOutput(self, context, state, unused, output):
        output[:] = 0

class ValkyriePDController(VectorSystem):
    """
    A simple PD controller that regulates the robot to a nominal (standing) position.
    """
    def __init__(self, tree, Kp=1000, Kd=1):
        VectorSystem.__init__(self, 
                              tree.get_num_positions() + tree.get_num_velocities(),   # input size [q,qd]
                              tree.get_num_actuators())   # output size [tau]

        self.np = tree.get_num_positions()
        self.nv = tree.get_num_velocities()
        self.nu = tree.get_num_actuators()

        self.nominal_state = RPYValkyrieFixedPointState()
        self.tau_ff = RPYValkyrieFixedPointTorque()

        self.tree = tree   # rigid body tree describing this robot

        self.Kp = Kp
        self.Kd = Kd

    def StateToQV(self, state):
        """
        Given a full state vector [q, v], extract q and v.
        """
        q = state[:self.tree.get_num_positions()]
        v = state[self.tree.get_num_positions():]

        return (q,v)

    def ComputePDControl(self,q,qd,feedforward=True):
        """
        Map state [q,qd] to control inputs [u].
        """
        q_nom, qd_nom = self.StateToQV(self.nominal_state)

        # compute torques to be applied in joint space
        if feedforward:
            tau = self.tau_ff + self.Kp*(q_nom-q) + self.Kd*(qd_nom - qd)
        else:
            tau = self.Kp*(q_nom-q) + self.Kd*(qd_nom - qd)

        # Convert torques to actuator space
        B = self.tree.B
        u = np.matmul(B.T,tau)

        return u

    def DoCalcVectorOutput(self, context, state, unused, output):
        q,qd = self.StateToQV(state)
        output[:] = self.ComputePDControl(q,qd,feedforward=True)
        

class ValkyrieController(ValkyriePDController):
    def __init__(self, tree):
        ValkyriePDController.__init__(self, 
                                      tree)

        self.mu = 0.3   # friction coefficient

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

    def get_contact_jacobians(self, cache, support="double"):
        """
        Return a list of contact jacobians for the given support phase (double, right, or left). 
        Assumes that each foot has corner contacts as defined by self.get_foot_contact_points().
        """
        world_index = self.tree.world().get_body_index()
        right_foot_index = self.tree.FindBody('rightFoot').get_body_index()
        left_foot_index = self.tree.FindBody('leftFoot').get_body_index()
        
        if support == "double":
            feet = [right_foot_index, left_foot_index]
        elif support == "right":
            feet = [right_foot_index]
        elif support == "left":
            feet = [left_foot_index]

        contact_jacobians = []

        for foot_index in feet:
            for contact_point in self.get_foot_contact_points():
                J = self.tree.transformPointsJacobian(cache,             # kinematics cache
                                                      contact_point,     # point in foot frame
                                                      foot_index,        # foot frame index
                                                      world_index,       # world frame index
                                                      False)             # in terms of qd as opposed to v
                                                                      
                contact_jacobians.append(J)

        return contact_jacobians

    def get_swing_foot_jacobian(self, cache, foot_index):
        """
        For the given foot, compute the jacobian J and time derivative Jd*qd.
        """
        world_index = self.tree.world().get_body_index()
        J = self.tree.transformPointsJacobian(cache, [0,0,0], foot_index, world_index, False)
        Jd_qd = self.tree.transformPointsJacobianDotTimesV(cache,[0,0,0], foot_index, world_index)

        return J, Jd_qd

    def solve_QP(self, cache, contact_jacobians, J_foot, Jd_qd_foot, xdd_foot_des, xdd_com_des, qdd_des):
        """
        Solve a quadratic program which attempts to regulate the joints to the desired
        accelerations and center of mass to the desired position as follows:
            min   w_1*| J_com*qdd + J_comd*qd - xdd_com_des |^2 + w_2*| qdd_des - qdd |^2
            s.t.  H*qdd + C = B*tau + sum(J'*f)
                  f \in friction cones
                  J*qdd + Jd*qd == 0  for all contacts
        """
        mp = MathematicalProgram()
        num_contacts = len(contact_jacobians)

        # Set weightings for prioritized task-space control
        w1 = 1.0   # center-of-mass tracking
        w2 = 0.1   # joint tracking
        w3 = 1.5   # foot tracking

        # Get dynamic quantities
        H = self.tree.massMatrix(cache)
        C = self.tree.dynamicsBiasTerm(cache, {}, True)
        B = self.tree.B

        J_com = self.tree.centerOfMassJacobian(cache)
        Jd_qd_com = self.tree.centerOfMassJacobianDotTimesV(cache)

        # create optimization variables
        qdd = mp.NewContinuousVariables(self.nv, 'qdd')   # joint accelerations
        tau = mp.NewContinuousVariables(self.nu, 'tau')   # applied torques
       
        f_contact = {}
        for i in range(num_contacts):        # contact forces
            f_contact[i] = mp.NewContinuousVariables(3, 'f_%s'%i)

        # Cast vectors as nx1 numpy arrays to allow for matrix multiplication with np.dot()
        C = C[np.newaxis].T
        Jd_qd_com = Jd_qd_com[np.newaxis].T
        xdd_com_des = xdd_com_des[np.newaxis].T
        Jd_qd_foot = Jd_qd_foot[np.newaxis].T
        xdd_foot_des = xdd_foot_des[np.newaxis].T
        tau = tau[np.newaxis].T
        qdd = qdd[np.newaxis].T
        qdd_des = qdd_des[np.newaxis].T

        # Center of mass tracking cost
        x_com_err = np.dot(J_com,qdd) + Jd_qd_com - xdd_com_des
        x_com_cost = np.dot(x_com_err.T,x_com_err)    # this is a 1x1 np.ndarray and we need a pydrake.symbolic.Expression
        x_com_cost = x_com_cost[0,0]                  # to use as a cost, so we simply extract the first element.

        mp.AddQuadraticCost(w1*x_com_cost)
       
        # Joint tracking cost
        q_err = qdd - qdd_des
        q_cost = np.dot(q_err.T,q_err)
        q_cost = q_cost[0,0]

        mp.AddQuadraticCost(w2*q_cost)

        # Foot tracking cost
        x_foot_err = np.dot(J_foot,qdd) + Jd_qd_foot - xdd_foot_des
        x_foot_cost = np.dot(x_foot_err.T,x_foot_err)[0,0]

        mp.AddQuadraticCost(w3*x_foot_cost)
       
        # Dynamic constraints 
        f_ext = sum([np.dot(contact_jacobians[i].T, f_contact[i][np.newaxis].T) for i in range(num_contacts)])
        lhs = np.dot(H,qdd) + C
        rhs = np.dot(B,tau) + f_ext

        for i in range(self.nv):
            mp.AddLinearConstraint(lhs[i,0] == rhs[i,0])

        # Friction cone (really pyramid) constraints 
        for j in range(num_contacts):
            mp.AddLinearConstraint(f_contact[j][0] + f_contact[j][1] <= self.mu*f_contact[j][2])
            mp.AddLinearConstraint(-f_contact[j][0] + f_contact[j][1] <= self.mu*f_contact[j][2])
            mp.AddLinearConstraint(f_contact[j][0] - f_contact[j][1] <= self.mu*f_contact[j][2])
            mp.AddLinearConstraint(-f_contact[j][0] - f_contact[j][1] <= self.mu*f_contact[j][2])

        # Solve the QP
        result = Solve(mp)

        if result.is_success():
            return result.GetSolution(tau)
        else:
            print("Whole-body QP Solver Failed! Falling back to PD controller")
            return self.ComputePDControl(q,qd,feedforward=True)

    

    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        q, qd = self.StateToQV(state)

        # Run kinematics, which will allow us to calculate key quantities
        cache = self.tree.doKinematics(q,qd)

        # Compute contact jacobians
        if context.get_time() <= 1:
            contact_jacobians = self.get_contact_jacobians(cache, support="double")
        else:
            contact_jacobians = self.get_contact_jacobians(cache, support="left")

        # Compute desired joint acclerations
        q_nom, qd_nom = self.StateToQV(self.nominal_state)
        qdd_des = 1*(q_nom-q) + 10*(qd_nom-qd)

        # Compute desired center of mass acceleration
        x_com = self.tree.centerOfMass(cache)
        xd_com = np.dot(self.tree.centerOfMassJacobian(cache), qd)
        x_com_nom, xd_com_nom = ComTrajectory(context.get_time())

        xdd_com_des = 100*(x_com_nom-x_com) + 10*(xd_com_nom-xd_com)

        # Compute desired swing (right) foot acceleration
        swing_foot_index = self.tree.FindBody('rightFoot').get_body_index()
        J_foot, Jd_qd_foot = self.get_swing_foot_jacobian(cache,swing_foot_index)
        x_foot = self.tree.transformPoints(cache,[0,0,0],swing_foot_index,0).flatten()
        xd_foot = np.dot(J_foot,qd)
        x_foot_nom, xd_foot_nom = FootTrajectory(context.get_time())
        
        xdd_foot_des = 100*(x_foot_nom-x_foot) + 1*(xd_foot_nom-xd_foot)

        start_time = time.time()
        # Solve QP to get desired torques
        u = self.solve_QP(cache, contact_jacobians, J_foot, Jd_qd_foot, xdd_foot_des, xdd_com_des, qdd_des)
        print(time.time()-start_time)

        output[:] = u
