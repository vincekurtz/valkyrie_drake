##
#
# Controller using the approximate simulation relation between the LIPM and
# the centroidal dynamics.
#
##

import itertools
from qp_controller import *

class ValkyrieASController(ValkyrieQPController):
    def __init__(self, tree, plant, dt):
        ValkyrieQPController.__init__(self, tree, plant)

        # Timestepping parameter for MPC and simulating the LIPM.
        self.dt = dt

        # Finite state machine defining desired ZMP trajectory,
        # foot placements, and swing foot trajectories.
        #self.fsm = WalkingFSM(n_steps=4,
        #                      step_length=0.5,
        #                      step_height=0.05,
        #                      step_time=0.9)
        self.fsm = StandingFSM()

        # Abstract Model Dynamics
        #
        #   x2_dot = A2*x2 + B2*u2
        #
        # where x2 is a desired CoM position and u2 is a desired
        # CoM velocity.
        self.A2 = np.zeros((3,3))
        self.B2 = np.eye(3)

        # Initial abstract system state
        self.x2 = np.array([0.0,0.0,0.96]).reshape(3,1)

        # Stuff to record for later plotting
        self.t = []                 # timesteps
        self.V = []                 # simulation function
        self.err = []               # output error
        self.y1 = np.empty((3,1))   # concrete system output: true CoM position
        self.y2 = np.empty((3,1))   # abstract system output: desired CoM position

    def get_foot_contact_points(self):
        """
        Return a tuple of points in the foot frame that represent contact locations. 
        (this is a rough guess based on self.tree.getTerrainContactPoints)
        """
        corner_contacts = (
                            np.array([-0.047, 0.07, -0.1]),  # slight inner approximation
                            np.array([-0.047,-0.07, -0.1]),
                            np.array([ 0.19,-0.07, -0.1]),
                            np.array([ 0.19, 0.07, -0.1])
                          )
        return corner_contacts


    def AddInterfaceConstraint(self, S, contact_jacobians, contact_forces, N, A_int, b_int, u2, tau, tau0):
        """
        Add the interface constraint

            S'tau + sum(J_contact'f_contact) = uV(x1,x2,u2) + N'*tau0

        where

            uV(x1,x2,u2) = A_int*u2 + b_int

        to the whole-body QP as a linear constraint on tau, f_contact, u2, and tau0.
        """
        contact_jacobians_transpose = np.hstack([contact_jacobians[i].T for i in range(len(contact_jacobians))])

        A_iface_eq = np.block([S.T, -N.T, -A_int, contact_jacobians_transpose])
        b_iface_eq = b_int
        vars_iface = np.vstack([tau, tau0, u2] + contact_forces)

        self.mp.AddLinearEqualityConstraint(A_iface_eq, b_iface_eq, vars_iface)

    def SolveWholeBodyQP(self, cache, context, q, qd, u2_nom):
        """
        Formulates and solves a quadratic program which enforces an approximate
        simulation-based interface uV(x1,x2,u2) while ensuring contact constraints
        are met:

            minimize:
                w1* || u2 - u2_nom ||^2 +
                w2* || qdd_des - qdd ||^2 +
                w3* || J_foot*qdd + Jd_foot*qd - xdd_foot_des ||^2 +
                w4* || J_torso*qdd + Jd_torso*qd - rpydd_torso_des ||^2
            subject to:
                    M*qdd + Cv + tau_g = S'*tau + sum(J'*f)
                    S'*tau + sum(J'*f) = uV(x1,x2,u2) + N'*tau_0
                    f \in friction cones
                    J_cj*qdd + J'_cj*qd == nu
                    nu_min <= nu <= nu_max

        Returns:
            tau - control torques to apply to the robot
            u2 - control input for the abstract system
        """
        

        ############## Tuneable Paramters ################

        w1 = 1e4    # abstract model input weight
        w2 = 0.1    # joint tracking weight
        w3 = 9e3   # foot tracking weight
        w4 = 50.0   # torso orientation weight
        w5 = 0.1    # centroidal momentum weight

        kappa = 500      # Interface PD gains
        Kd_int = 300

        nu_min = -1e-10   # slack for contact constraint
        nu_max = 1e-10

        Kp_q = 100     # Joint angle PD gains
        Kd_q = 10

        Kp_foot = 500.0   # foot position PD gains
        Kd_foot = 400.0 

        Kp_torso = 500.0   # torso orientation PD gains
        Kd_torso = 50.0

        Kp_k = 10.0    # angular momentum P gain

        Kd_contact = 200.0  # Contact movement damping P gain

        ##################################################
        
        # Specify support phase
        support = self.fsm.SupportPhase(context.get_time())

        # Compute dynamic quantities. Note that we cast vectors as nx1 numpy arrays to allow
        # for matrix multiplication with np.dot().
        M = self.tree.massMatrix(cache)
        tau_g = self.tree.dynamicsBiasTerm(cache,{},False).reshape(self.np,1)
        Cv = self.tree.dynamicsBiasTerm(cache,{},True).reshape(self.np,1) - tau_g
        S = self.tree.B.T

        # Center of mass jacobian
        J_com = self.tree.centerOfMassJacobian(cache)
        Jd_qd_com = self.tree.centerOfMassJacobianDotTimesV(cache)

        # Nullspace projector 
        Minv = np.linalg.inv(M)
        Lambda = np.linalg.inv( np.dot(np.dot(J_com,Minv),J_com.T))
        Jbar = np.dot( np.dot(Minv, J_com.T), Lambda)
        N = (np.eye(self.np) - np.dot(J_com.T,Jbar.T)).T

        # CoM angular momentum jacobian
        J_k = self.tree.centroidalMomentumMatrix(cache)[0:3,:]
        Jd_qd_k = self.tree.centroidalMomentumMatrixDotTimesV(cache)[0:3]
        
        # Foot Jacobians
        J_left, Jd_qd_left = self.get_body_jacobian(cache, self.left_foot_index)
        J_right, Jd_qd_right = self.get_body_jacobian(cache, self.right_foot_index)

        J_rpyleft = self.tree.relativeRollPitchYawJacobian(cache, self.world_index, self.left_foot_index, True)
        Jd_qd_rpyleft = self.tree.relativeRollPitchYawJacobianDotTimesV(cache,
                                                                        self.world_index,
                                                                        self.left_foot_index)[np.newaxis].T
        
        J_rpyright = self.tree.relativeRollPitchYawJacobian(cache, self.world_index, self.right_foot_index, True)
        Jd_qd_rpyright = self.tree.relativeRollPitchYawJacobianDotTimesV(cache,
                                                                        self.world_index,
                                                                        self.right_foot_index)[np.newaxis].T

        # Contact point jacobians
        contact_jacobians, contact_jacobians_dot_v = self.get_contact_jacobians(cache, support)
        num_contacts = len(contact_jacobians)

        # torso jacobian
        J_torso = self.tree.relativeRollPitchYawJacobian(cache, self.world_index, self.torso_index, True)
        Jd_qd_torso = self.tree.relativeRollPitchYawJacobianDotTimesV(cache,
                                                                      self.world_index,
                                                                      self.torso_index)[np.newaxis].T

        # Compute desired joint acclerations
        q_nom = self.nominal_state[0:self.np]
        qd_nom = self.nominal_state[self.np:]
        qdd_des = Kp_q*(q_nom-q) + Kd_q*(qd_nom-qd)
        qdd_des = qdd_des[np.newaxis].T

        # Compute desired linear accelerations of the feet
        x_left = self.tree.transformPoints(cache, [0,0,0], self.left_foot_index, self.world_index)
        xd_left = np.dot(J_left, qd)[np.newaxis].T
        x_left_nom, xd_left_nom = self.fsm.LeftFootTrajectory(context.get_time())
        xdd_left_des = Kp_foot*(x_left_nom - x_left) + Kd_foot*(xd_left_nom - xd_left)
        
        x_right = self.tree.transformPoints(cache, [0,0,0], self.right_foot_index, self.world_index)
        xd_right = np.dot(J_right, qd)[np.newaxis].T
        x_right_nom, xd_right_nom = self.fsm.RightFootTrajectory(context.get_time())
        xdd_right_des = Kp_foot*(x_right_nom - x_right) + Kd_foot*(xd_right_nom - xd_right)

        # Compute desired angular accelerations of the feet
        rpy_left = self.tree.relativeRollPitchYaw(cache, self.world_index, self.left_foot_index)[np.newaxis].T
        rpyd_left = np.dot(J_rpyleft,qd)[np.newaxis].T
        rpy_left_nom = np.asarray([[0.0],[0.0],[0.0]])
        rpyd_left_nom = np.zeros((3,1))
        rpydd_left_des = Kp_foot*(rpy_left_nom - rpy_left) + Kd_foot*(rpyd_left_nom - rpyd_left)

        rpy_right = self.tree.relativeRollPitchYaw(cache, self.world_index, self.right_foot_index)[np.newaxis].T
        rpyd_right = np.dot(J_rpyright,qd)[np.newaxis].T
        rpy_right_nom = np.asarray([[0.0],[0.0],[0.0]])
        rpyd_right_nom = np.zeros((3,1))
        rpydd_right_des = Kp_foot*(rpy_right_nom - rpy_right) + Kd_foot*(rpyd_right_nom - rpyd_right)

        # Compute desired torso angular acceleration
        rpy_torso = self.tree.relativeRollPitchYaw(cache, self.world_index, self.torso_index)[np.newaxis].T
        rpyd_torso = np.dot(J_torso,qd)[np.newaxis].T
        rpy_torso_nom = np.asarray([[0.0],[-0.1],[0.0]])
        rpyd_torso_nom = np.zeros((3,1))

        rpydd_torso_des = Kp_torso*(rpy_torso_nom - rpy_torso) + Kd_torso*(rpyd_torso_nom - rpyd_torso)

        # Compute desired CoM angular momentum dot
        k_com = np.dot(J_k, qd)[np.newaxis].T
        k_com_nom = np.zeros((3,1))
        kd_com_des = Kp_k*(k_com_nom - k_com)
        
        # Compute energy shaping-based interface as a linear constraint 
        # uV = tau_g - kappa*J'*(x_task-x2) - Kd*(qd-Jbar*u2) = A_int*u2 + b_int
        x_task = self.tree.centerOfMass(cache)[np.newaxis].T
        Jbar_com = np.dot( np.linalg.inv( np.dot(J_com.T,J_com) + 1e-8*np.eye(self.np)), J_com.T)
        A_int = Kd_int*Jbar_com
        b_int = tau_g - kappa*np.dot(J_com.T, x_task-self.x2) - Kd_int*qd.reshape(self.nv,1)

        #################### QP Formulation ##################

        self.mp = MathematicalProgram()
        
        # create optimization variables
        qdd = self.mp.NewContinuousVariables(self.nv, 1, 'qdd')   # joint accelerations
        tau = self.mp.NewContinuousVariables(self.nu, 1, 'tau')   # applied torques

        u2 = self.mp.NewContinuousVariables(len(u2_nom), 1, 'u2') # input to abstract system
        tau0 = self.mp.NewContinuousVariables(self.np, 1, 'tau0') # (fictional) torques in nullspace of CoM motion
       
        f_contact = [self.mp.NewContinuousVariables(3,1,'f_%s'%i) for i in range(num_contacts)]

        # Nominal u2 tracking cost
        u2_track_cost = self.mp.AddQuadraticErrorCost(Q=w1*np.eye(3),x_desired=u2_nom,vars=u2)
       
        # Joint tracking cost
        joint_cost = self.mp.AddQuadraticErrorCost(Q=w2*np.eye(self.nv),x_desired=qdd_des,vars=qdd)

        # Foot tracking costs: position and orientation for each foot
        self.AddJacobianTypeCost(J_left, qdd, Jd_qd_left, xdd_left_des, weight=w3)
        self.AddJacobianTypeCost(J_rpyleft, qdd, Jd_qd_rpyleft, rpydd_left_des, weight=w3)

        self.AddJacobianTypeCost(J_right, qdd, Jd_qd_right, xdd_right_des, weight=w3)
        self.AddJacobianTypeCost(J_rpyright, qdd, Jd_qd_rpyright, rpydd_right_des, weight=w3)
        
        # torso orientation cost
        torso_cost = self.AddJacobianTypeCost(J_torso, qdd, Jd_qd_torso, rpydd_torso_des, weight=w4)

        # angular momentum
        angular_cost = self.AddJacobianTypeCost(J_k, qdd, Jd_qd_k, kd_com_des, weight=w5)
            
        # Contact acceleration constraint
        for j in range(num_contacts):
            J_cont = contact_jacobians[j]
            Jd_qd_cont = contact_jacobians_dot_v[j]
            xd_cont = np.dot(J_cont,qd)[np.newaxis].T  # (3x1) vector
            xdd_cont_des = -Kd_contact*xd_cont

            contact_constraint = self.AddJacobianTypeConstraint(J_cont, qdd, Jd_qd_cont, xdd_cont_des)
 
            # add some slight flexibility to this constraint to avoid infeasibility
            contact_constraint.evaluator().UpdateUpperBound(nu_max*np.array([1.0,1.0,1.0]))
            contact_constraint.evaluator().UpdateLowerBound(nu_min*np.array([1.0,1.0,1.0]))
       
        # Dynamic constraints 
        dynamics_constraint = self.AddDynamicsConstraint(M, qdd, Cv+tau_g, S.T, tau, contact_jacobians, f_contact)

        # Add interface constraint S'*tau + sum(J'*f_ext) = uV + N'*tau_0
        self.AddInterfaceConstraint(S, contact_jacobians, f_contact, N, A_int, b_int, u2, tau, tau0)

        # Friction cone (really pyramid) constraints 
        friction_constraint = self.AddFrictionPyramidConstraint(f_contact)

        # Solve the QP
        result = Solve(self.mp)

        assert result.is_success(), "Whole-body QP Failed"

        return (result.GetSolution(tau), result.GetSolution(u2))

    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """
        q, qd = self.StateToQQDot(state)

        # Compute kinimatics, which will allow us to calculate key quantities
        cache = self.tree.doKinematics(q,qd)

        # Comput nominal input to abstract system (CoM velocity)
        x2_nom = np.array([0.11, 0.0, 0.96]).reshape(3,1)
        u2_nom = -1.0*(self.x2 - x2_nom)
        #u2_nom = np.array([0.05, 0.0, 0.0]).reshape(3,1)  # just try to drive CoM forward

        tau, u2 = self.SolveWholeBodyQP(cache, context, q, qd, u2_nom)

        # Set control torques to be sent to the robot
        output[:] = tau

        # Simulate abstract system forward in time
        x2_dot = np.dot(self.A2,self.x2) + np.dot(self.B2,u2.reshape(3,1))
        self.x2 = self.x2 + x2_dot*self.dt

        # Record stuff for later plotting
        self.t.append(context.get_time())

        y1 = self.tree.centerOfMass(cache).reshape(3,1)
        y2 = self.x2
        self.y1 = np.hstack([self.y1, y1])
        self.y2 = np.hstack([self.y2, y2])

        err = np.dot((y1-y2).T,(y1-y2))[0,0]
        self.err.append(err)

        M = self.tree.massMatrix(cache)
        J = self.tree.centerOfMassJacobian(cache)
        kappa = 500
        V = 0.5*np.dot(np.dot(qd.T,M),qd) + 500*err
        self.V.append(V)

