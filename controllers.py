##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import numpy as np
import time
from pydrake.all import *
from casadi import *
from helpers import *

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

        self.last_iteration_solution = None
        self.last_iteration_duals = None

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

    def get_contact_jacobians(self, cache, support="double"):
        """
        Return a list of contact jacobians for both feet, assuming double support and that
        each foot has contacts in the corners, as defined by self.get_foot_contact_points()
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
                                                      False)             # in terms of qd
                                                                      
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

        assert result.is_success(), "Whole-body QP Solver Failed!"

        return result.GetSolution(tau)
    
    def solve_QP_casadi(self, cache, contact_jacobians, J_foot, Jd_qd_foot, xdd_foot_des, xdd_com_des, qdd_des):
        """
        Solve a quadratic program which attempts to regulate the joints to the desired
        accelerations and center of mass to the desired position as follows:
            min   w_1*| J_com*qdd + J_comd*qd - xdd_com_des |^2 + w_2*| qdd_des - qdd |^2
            s.t.  H*qdd + C = B*tau + sum(J'*f)
                  f \in friction cones
                  J*qdd + Jd*qd == 0  for all contacts
        """
        opti = casadi.Opti()
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
        qdd = opti.variable(self.nv)    # joint accelerations
        tau = opti.variable(self.nu)    # applied torques
       
        f_contact = {}
        for i in range(num_contacts):        # contact forces
            f_contact[i] = opti.variable(3)

        # Center of mass tracking cost
        x_com_err = mtimes(J_com,qdd) + Jd_qd_com - xdd_com_des
        x_com_cost = w1*dot(x_com_err, x_com_err)
       
        # Joint tracking cost
        q_err = qdd - qdd_des
        q_cost = w2*dot(q_err,q_err)

        # Foot tracking cost
        x_foot_err = mtimes(J_foot,qdd) + Jd_qd_foot - xdd_foot_des
        x_foot_cost = w3*dot(x_foot_err,x_foot_err)

        opti.minimize(x_com_cost + q_cost + x_foot_cost)
       
        # Dynamic constraints 
        f_ext = sum([mtimes(contact_jacobians[i].T, f_contact[i]) for i in range(num_contacts)])
        
        opti.subject_to( mtimes(H,qdd) + C == mtimes(B,tau) + f_ext )

        # Friction cone (really pyramid) constraints 
        for j in range(num_contacts):
            opti.subject_to(f_contact[j][0] + f_contact[j][1] <= self.mu*f_contact[j][2])
            opti.subject_to(-f_contact[j][0] + f_contact[j][1] <= self.mu*f_contact[j][2])
            opti.subject_to(f_contact[j][0] - f_contact[j][1] <= self.mu*f_contact[j][2])
            opti.subject_to(-f_contact[j][0] - f_contact[j][1] <= self.mu*f_contact[j][2])

        # Warm-start if possible
        if self.last_iteration_solution is not None:
            opti.set_initial(self.last_iteration_solution)
            opti.set_initial(opti.lam_g, self.last_iteration_duals)

        # Solve the QP
        options = {"ipopt.print_level":0, "print_time": 0}  # supress solver output
        opti.solver('ipopt',options)
        sol = opti.solve()

        # Save optimal values for next iteration
        self.last_iteration_solution = sol.value_variables()
        self.last_iteration_duals = sol.value(opti.lam_g)

        return sol.value(tau)


    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        q = state[:self.np]
        qd = state[self.np:]

        # Run kinematics, which will allow us to calculate the dynamics
        cache = self.tree.doKinematics(q,qd)

        # Compute contact jacobians
        if context.get_time() <= 1:
            contact_jacobians = self.get_contact_jacobians(cache, support="double")
        else:
            contact_jacobians = self.get_contact_jacobians(cache, support="left")

        # Compute desired joint acclerations
        q_nom = self.nominal_state[:self.np]
        qd_nom = self.nominal_state[self.np:]

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
        tau_qp = self.solve_QP(cache, contact_jacobians, J_foot, Jd_qd_foot, xdd_foot_des, xdd_com_des, qdd_des)
        print(time.time()-start_time)

        output[:] = tau_qp
