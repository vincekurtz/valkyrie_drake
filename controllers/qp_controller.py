##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

from pd_controller import *

class ValkyrieQPController(ValkyriePDController):
    def __init__(self, tree, plant):
        ValkyriePDController.__init__(self, tree, plant)

        self.fsm = WalkingFSM(n_steps=8,         # Finite State Machine describing CoM trajectory,
                              step_length=0.50,   # swing foot trajectories, and stance phases.
                              step_height=0.10,
                              step_time=1.0)
        #self.fsm = StandingFSM()

        self.mu = 0.1             # assumed friction coefficient

        self.mp = MathematicalProgram()  # QP that we'll use for whole-body control

        self.right_foot_index = self.tree.FindBody('rightFoot').get_body_index()
        self.left_foot_index = self.tree.FindBody('leftFoot').get_body_index()
        self.world_index = self.tree.world().get_body_index()
        self.torso_index = self.tree.FindBody('torso').get_body_index()
        
        # Stuff to record for later plotting
        self.t = []                 # timesteps
        self.V = []                 # simulation function (not relevant here, just so plots don't break)
        self.err = []               # output error
        self.y1 = np.empty((3,1))   # concrete system output: true CoM position
        self.y2 = np.empty((3,1))   # abstract system output: desired CoM position
        self.tau = np.empty((self.nu,1))  # applied control torques

    def get_foot_contact_points(self):
        """
        Return a tuple of points in the foot frame that represent contact locations. 
        (this is a rough guess based on self.tree.getTerrainContactPoints)
        """
        corner_contacts = (
                            np.array([-0.07, 0.08, -0.1]),
                            np.array([-0.07,-0.08, -0.1]),
                            np.array([ 0.20,-0.08, -0.1]),
                            np.array([ 0.20, 0.08, -0.1])
                          )
        return corner_contacts

    def get_contact_jacobians(self, cache, support="double"):
        """
        Return a list of contact jacobians for the given support phase (double, right, or left). 
        Assumes that each foot has corner contacts as defined by self.get_foot_contact_points().
        """
        
        if support == "double":
            feet = [self.right_foot_index, self.left_foot_index]
        elif support == "right":
            feet = [self.right_foot_index]
        elif support == "left":
            feet = [self.left_foot_index]

        contact_jacobians = []
        contact_jacobians_dot_v = []

        for foot_index in feet:
            for contact_point in self.get_foot_contact_points():
                J = self.tree.transformPointsJacobian(cache,             # kinematics cache
                                                      contact_point,     # point in foot frame
                                                      foot_index,        # foot frame index
                                                      self.world_index,  # world frame index
                                                      False)             # in terms of qd as opposed to v

                Jd_v = self.tree.transformPointsJacobianDotTimesV(cache,
                                                                  contact_point,
                                                                  foot_index,
                                                                  self.world_index)
                                                                      
                contact_jacobians.append(J)
                contact_jacobians_dot_v.append(Jd_v[np.newaxis].T)

        return contact_jacobians, contact_jacobians_dot_v

    def get_body_jacobian(self, cache, body_index, relative_position=[0,0,0]):
        """
        For the given body index, compute the jacobian J and time derivative Jd*qd in the
        world frame. 
        """
        J = self.tree.transformPointsJacobian(cache, relative_position, body_index, self.world_index, False)
        Jd_qd = self.tree.transformPointsJacobianDotTimesV(cache, relative_position, body_index, self.world_index)

        # Cast vector as 1xn numpy array
        Jd_qd = Jd_qd[np.newaxis].T

        return J, Jd_qd
    
    def AddJacobianTypeCost(self, J, qdd, Jd_qd, xdd_des, weight=1.0):
        """
        Add a quadratic cost of the form 
            weight*| J*qdd + Jd_qd - xdd_des |^2
        to the whole-body controller QP.
        """
        # Put in the form 1/2*qdd'*Q*qdd + c'*qdd for fast formulation
        Q = weight*np.dot(J.T,J)
        c = weight*(np.dot(Jd_qd.T,J) - np.dot(xdd_des.T,J)).T

        return self.mp.AddQuadraticCost(Q,c,qdd)

    def AddJacobianTypeConstraint(self, J, qdd, Jd_qd, xdd_des):
        """
        Add a linear constraint of the form
            J*qdd + Jd_qd == xdd_des
        to the whole-body controller QP.
        """
        A_eq = J     # A_eq*qdd == b_eq
        b_eq = xdd_des-Jd_qd

        return self.mp.AddLinearEqualityConstraint(A_eq, b_eq, qdd)

    def AddDynamicsConstraint(self, H, qdd, C, B, tau, contact_jacobians, f_contact):
        """
        Add a dynamics constraint of the form 
            H*qdd + C == B*tau + sum(J[i]'*f_contact[i])
        to the whole-body controller QP.
        """
        # We'll rewrite the constraints in the form A_eq*x == b_eq for speed
        A_eq = np.hstack([H, -B])
	x = np.vstack([qdd,tau])

	for i in range(len(contact_jacobians)):
	    A_eq = np.hstack([A_eq, -contact_jacobians[i].T])
	    x = np.vstack([x,f_contact[i]])

	b_eq = -C

	return self.mp.AddLinearEqualityConstraint(A_eq,b_eq,x)

    def AddFrictionPyramidConstraint(self, f_contact):
        """
        Add a friction pyramid constraint for the given set of contact forces
        to the whole-body controller QP. 
        """
        num_contacts = len(f_contact)

        A_i = np.asarray([[ 1, 0, -self.mu],   # pyramid approximation of CWC for one
                          [-1, 0, -self.mu],   # contact force f \in R^3
                          [ 0, 1, -self.mu],
                          [ 0,-1, -self.mu]])

        # We'll formulate as lb <= Ax <= ub, where x=[f_1',f_2',...]'
        A = np.kron(np.eye(num_contacts),A_i)

        ub = np.zeros((4*num_contacts,1))
        lb = -np.inf*np.ones((4*num_contacts,1))

        x = np.vstack([f_contact[i] for i in range(num_contacts)])

        return self.mp.AddLinearConstraint(A=A,lb=lb,ub=ub,vars=x)


    def SolveWholeBodyQP(self, cache, context, q, qd):
        """
        Formulates and solves a quadratic program which attempts to regulate the joints to the desired
        accelerations and center of mass to the desired position as follows:

        minimize:
            w_1* || J_com*qdd + Jd_com*qd - xdd_com_des ||^2 + 
            w_2* || A*qdd + Ad*qd - hd_com_des ||^2 +
            w_3* || qdd_des - qdd ||^2 +
            w_4* || J_left*qdd + Jd_left*qd - xdd_left_des ||^2 +
            w_4* || J_right*qdd + Jd_right*qd - xdd_right_des ||^2 +
            w_5* || J_torso*qdd + Jd_torso*qd - rpydd_torso_des ||^2
        subject to:
                H*qdd + C = B*tau + sum(J'*f)
                f \in friction cones
                J_cj*qdd + J'_cj*qd == nu
                nu_min <= nu <= nu_max
        """

        ############## Tuneable Paramters ################

        w1 = 50.0   # center-of-mass tracking weight
        w2 = 0.1    # centroid momentum weight
        w3 = 0.5    # joint tracking weight
        w4 = 50.0   # foot tracking weight
        w5 = 50.0   # torso orientation weight

        nu_min = -1e-10   # slack for contact constraint
        nu_max = 1e-10

        Kp_q = 100     # Joint angle PD gains
        Kd_q = 10

        Kp_com = 1e4   # Center of mass PD gains
        Kd_com = 1000.0

        Kp_h = 10.0    # Centroid momentum P gain

        Kp_foot = 500.0   # foot position PD gains
        Kd_foot = 50.0 

        Kp_torso = 500.0   # torso orientation PD gains
        Kd_torso = 50.0

        Kd_contact = 10.0  # Contact movement damping P gain

        ##################################################
        
        # Specify support phase
        support = self.fsm.SupportPhase(context.get_time())

        # Compute dynamic quantities. Note that we cast vectors as nx1 numpy arrays to allow
        # for matrix multiplication with np.dot().
        H = self.tree.massMatrix(cache)                  # Equations of motion
        C = self.tree.dynamicsBiasTerm(cache, {}, True)[np.newaxis].T
        B = self.tree.B

        A = self.tree.centroidalMomentumMatrix(cache)
        Ad_qd = self.tree.centroidalMomentumMatrixDotTimesV(cache)

        J_com = self.tree.centerOfMassJacobian(cache)    # Center of mass jacobian
        Jd_qd_com = self.tree.centerOfMassJacobianDotTimesV(cache)[np.newaxis].T
       
        J_left, Jd_qd_left = self.get_body_jacobian(cache, self.left_foot_index)  # foot jacobians
        J_right, Jd_qd_right = self.get_body_jacobian(cache, self.right_foot_index)
        
        J_rpyleft = self.tree.relativeRollPitchYawJacobian(cache, self.world_index, self.left_foot_index, True)
        Jd_qd_rpyleft = self.tree.relativeRollPitchYawJacobianDotTimesV(cache,
                                                                        self.world_index,
                                                                        self.left_foot_index)[np.newaxis].T
        
        J_rpyright = self.tree.relativeRollPitchYawJacobian(cache, self.world_index, self.right_foot_index, True)
        Jd_qd_rpyright = self.tree.relativeRollPitchYawJacobianDotTimesV(cache,
                                                                        self.world_index,
                                                                        self.right_foot_index)[np.newaxis].T

        contact_jacobians, contact_jacobians_dot_v = self.get_contact_jacobians(cache, support)
        num_contacts = len(contact_jacobians)

        J_torso = self.tree.relativeRollPitchYawJacobian(cache, self.world_index, self.torso_index,True)
        Jd_qd_torso = self.tree.relativeRollPitchYawJacobianDotTimesV(cache,
                                                                      self.world_index,
                                                                      self.torso_index)[np.newaxis].T

        # Compute desired joint acclerations
        q_nom = self.nominal_state[0:self.np]
        qd_nom = self.nominal_state[self.np:]
        qdd_des = Kp_q*(q_nom-q) + Kd_q*(qd_nom-qd)
        qdd_des = qdd_des[np.newaxis].T

        # Compute desired center of mass acceleration
        x_com = self.tree.centerOfMass(cache)[np.newaxis].T
        xd_com = np.dot(J_com, qd)[np.newaxis].T
        x_com_nom, xd_com_nom, xdd_com_nom = self.fsm.ComTrajectory(context.get_time())

        xdd_com_des = xdd_com_nom + Kp_com*(x_com_nom-x_com) + Kd_com*(xd_com_nom-xd_com)
       
        # Compute desired centroid momentum dot
        h_com = np.dot(A, qd)[np.newaxis].T
        m = self.tree.massMatrix(cache)[0,0]  # total mass
        h_com_nom = np.vstack([np.zeros((3,1)),m*xd_com_nom])  # desired angular velocity is zero,
                                                               # CoM velocity matches the CoM trajectory
        hd_com_des = Kp_h*(h_com_nom - h_com)

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
        
        # Compute desired base frame angular acceleration
        rpy_torso = self.tree.relativeRollPitchYaw(cache, self.world_index, self.torso_index)[np.newaxis].T
        rpyd_torso = np.dot(J_torso,qd)[np.newaxis].T
        rpy_torso_nom = np.asarray([[0.0],[-0.1],[0.0]])
        rpyd_torso_nom = np.zeros((3,1))

        rpydd_torso_des = Kp_torso*(rpy_torso_nom - rpy_torso) + Kd_torso*(rpyd_torso_nom - rpyd_torso)


        #################### QP Formulation ##################

        self.mp = MathematicalProgram()
        
        # create optimization variables
        qdd = self.mp.NewContinuousVariables(self.nv, 1, 'qdd')   # joint accelerations
        tau = self.mp.NewContinuousVariables(self.nu, 1, 'tau')   # applied torques
       
        f_contact = [self.mp.NewContinuousVariables(3,1,'f_%s'%i) for i in range(num_contacts)]

        # Center of mass tracking cost
        com_cost = self.AddJacobianTypeCost(J_com, qdd, Jd_qd_com, xdd_com_des, weight=w1)

        # Centroidal momentum cost
        centroidal_cost = self.AddJacobianTypeCost(A, qdd, Ad_qd, hd_com_des, weight=w2)
       
        # Joint tracking cost
        joint_cost = self.mp.AddQuadraticErrorCost(Q=w3*np.eye(self.nv),x_desired=qdd_des,vars=qdd)
        
        # Foot tracking costs: position and orientation for each foot
        self.AddJacobianTypeCost(J_left, qdd, Jd_qd_left, xdd_left_des, weight=w4)
        self.AddJacobianTypeCost(J_rpyleft, qdd, Jd_qd_rpyleft, rpydd_left_des, weight=w4)

        self.AddJacobianTypeCost(J_right, qdd, Jd_qd_right, xdd_right_des, weight=w4)
        self.AddJacobianTypeCost(J_rpyright, qdd, Jd_qd_rpyright, rpydd_right_des, weight=w4)

        # torso orientation cost
        torso_cost = self.AddJacobianTypeCost(J_torso, qdd, Jd_qd_torso, rpydd_torso_des, weight=w5)
            
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
        dynamics_constraint = self.AddDynamicsConstraint(H, qdd, C, B, tau, contact_jacobians, f_contact)

        # Friction cone (really pyramid) constraints 
        friction_constraint = self.AddFrictionPyramidConstraint(f_contact)

        # Solve the QP
        result = Solve(self.mp)

        assert result.is_success()

        return result.GetSolution(tau)


    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """
        
        q, qd = self.StateToQQDot(state)

        # Run kinematics, which will allow us to calculate key quantities
        cache = self.tree.doKinematics(q, qd)

        # Solve the whole-body QP to get desired torques
        u = self.SolveWholeBodyQP(cache, context, q, qd)

        output[:] = u
        
        # Record stuff for later plotting
        self.t.append(context.get_time())

        y1 = self.tree.centerOfMass(cache).reshape(3,1)
        x_com_nom, _, _ = self.fsm.ComTrajectory(context.get_time())
        y2 = x_com_nom
        self.y1 = np.hstack([self.y1, y1])
        self.y2 = np.hstack([self.y2, y2])

        err = np.dot((y1-y2).T,(y1-y2))[0,0]
        self.err.append(err)

        self.V.append(err)     # placeholder for simulation function

        self.tau = np.hstack([self.tau,u.reshape(self.nu,1)])


