##
#
# Controller using the approximate simulation relation between the LIPM and
# the centroidal dynamics.
#
##

from qp_controller import *

class ValkyrieASController(ValkyrieQPController):
    def __init__(self, tree, plant, dt):
        ValkyrieQPController.__init__(self, tree, plant)

        # Timestepping parameter for MPC and simulating the LIPM.
        self.dt = dt

        # Finite state machine defining desired ZMP trajectory,
        # foot placements, and swing foot trajectories.
        self.fsm = StandingFSM()

        # Parameters
        h = 0.967
        g = 9.81
        omega = np.sqrt(g/h)
        m = get_total_mass(tree)

        # Define template (LIPM) dynamics
        self.A_lip = np.zeros((4,4))
        self.A_lip[0:2,2:4] = np.eye(2)
        self.A_lip[2:4,0:2] = omega**2*np.eye(2)

        self.B_lip = np.zeros((4,2))
        self.B_lip[2:4,:] = -omega**2*np.eye(2)

        self.C_lip = np.zeros((9,4))
        self.C_lip[0:2,0:2] = np.eye(2)
        self.C_lip[6:8,2:4] = m*np.eye(2)

        # Define anchor (centroidal) dynamics
        self.A_task = np.zeros((9,9))
        self.A_task[0:3,6:9] = 1/m*np.eye(3)

        self.B_task = np.zeros((9,6))
        self.B_task[3:9,:] = np.eye(6)

        self.C_task = np.eye(9)

        # Compute an interface that certifies an approximate simulation relation
        # between the template and the anchor
        interface_mp = MathematicalProgram()
        
        lmbda = 1.0
        Mbar = interface_mp.NewSymmetricContinuousVariables(9,"Mbar")

        Q_task = 1000*np.eye(9)
        R_task = np.eye(6)
        K, P = LinearQuadraticRegulator(self.A_task, self.B_task, Q_task, R_task)
        Kbar = np.dot(-K,Mbar)
        #Kbar = interface_mp.NewContinuousVariables(6,9,"Kbar")
   
        constraint_matrix_one = np.vstack([
                                    np.hstack([Mbar,                     np.dot(Mbar,self.C_task.T)]),
                                    np.hstack([np.dot(self.C_task,Mbar), np.eye(9)                 ])
                                ])
        interface_mp.AddPositiveSemidefiniteConstraint(constraint_matrix_one)

        constraint_matrix_two = -2*lmbda*Mbar - np.dot(Mbar,self.A_task.T) - np.dot(self.A_task,Mbar) \
                                        - np.dot(Kbar.T,self.B_task.T) - np.dot(self.B_task,Kbar)
        interface_mp.AddPositiveSemidefiniteConstraint(constraint_matrix_two)

        result = Solve(interface_mp)
        assert result.is_success()
        Mbar = result.GetSolution(Mbar)
        Kbar = result.GetSolution(Kbar)

        self.M = np.linalg.inv(Mbar)
        self.K = -K
        print(self.K)

        self.P = self.C_lip

        self.Q = np.zeros((6,4))
        self.Q[3:5,0:2] = omega**2*m*np.eye(2)

        self.R = np.zeros((6,2))
        self.R[3:5] = -m*omega**2*np.eye(2)

        # Double check the results
        #assert is_pos_def(self.M) , "M is not positive definite."
        assert is_pos_def(-self.A_task-np.dot(self.B_task,self.K)) , "A+BK is not Hurwitz"

        assert is_pos_def(self.M - np.dot(self.C_task.T,self.C_task)) , "Failed test M >= C'C"
        assert is_pos_def(-2*lmbda*self.M \
                          - np.dot((self.A_task+np.dot(self.B_task,self.K)).T,self.M) \
                          - np.dot(self.M,self.A_task+np.dot(self.B_task,self.K)) ) , "Failed test (A+BK)'M+M(A+BK) <= -2lmbdaM"

        assert np.all(self.C_lip == np.dot(self.C_task,self.P)) , "Failed test C_lip = C_task*P"

        assert np.all( np.dot(self.P,self.A_lip) == np.dot(self.A_task,self.P) + np.dot(self.B_task,self.Q) ) \
                    , "Failed Test P*A_lip = A_task*P+B*Q"

        # Set initial LIPM state
        self.x_lip = np.array([[0.0],   # x position
                               [0.0],   # y position
                               [0.0],   # x velocity
                               [0.0]])  # y velocity

    def GetTaskSpaceState(self, cache, q, qd):
        """
        Compute the current task-space state
        
            x_task = [ p_com
                       h_com ],

        where 
            p_com \in R^3 is the position of the center of mass and
            h_com \in R^6 is the centroidal momentum.
        """
        p_com = self.tree.centerOfMass(cache)[np.newaxis].T
        
        A_com = self.tree.centroidalMomentumMatrix(cache)
        h_com = np.dot(A_com, qd)[np.newaxis].T

        return np.vstack([p_com, h_com])

    def DoTemplateMPC(self, x_lip, x_task):
        """
        Given the current template (x_lip) and task space (x_task) states, perform MPC
        in the template model while respecting contact constraints for the full model.
        """
        u_lip = np.array([[0.0],
                          [0.0]])
        u_task = np.dot(self.R,u_lip) + np.dot(self.Q,x_lip) + np.dot(self.K, x_task-np.dot(self.P,x_lip))

        return u_lip, u_task

    def SolveWholeBodyQP(self, cache, time, q, qd, u_task):
        """
        Use a whole-body quadratic program to feedback linearize the task-space
        dynamics.
        """

        ############## Tuneable Paramters ################

        Kp_q = 100     # Joint angle PD gains
        Kd_q = 10

        Kp_com = 500   # Center of mass PD gains
        Kd_com = 50

        Kp_h = 10.0    # Centroid momentum P gain

        Kp_foot = 100.0   # foot position PD gains
        Kd_foot = 100.0 

        w1 = 50.0   # center-of-mass tracking weight
        w2 = 0.1  # centroid momentum weight
        w3 = 0.5   # joint tracking weight
        w4 = 50.0    # foot tracking weight

        nu_min = -0.001   # slack for contact constraint
        nu_max = 0.001

        ##################################################

        # Compute desired joint acclerations
        q_nom = self.nominal_state[0:self.np]
        qd_nom = self.nominal_state[self.np:]
        qdd_des = Kp_q*(q_nom-q) + Kd_q*(qd_nom-qd)
        qdd_des = qdd_des[np.newaxis].T

        # Compute desired center of mass acceleration
        x_com = self.tree.centerOfMass(cache)[np.newaxis].T
        xd_com = np.dot(self.tree.centerOfMassJacobian(cache), qd)[np.newaxis].T
        x_com_nom, xd_com_nom, xdd_com_nom = self.fsm.ComTrajectory(time)
        xdd_com_des = xdd_com_nom + Kp_com*(x_com_nom-x_com) + Kd_com*(xd_com_nom-xd_com)

        # Compute desired centroid momentum dot
        hd_com_des = u_task
        A_com = self.tree.centroidalMomentumMatrix(cache)
        Ad_com_qd = self.tree.centroidalMomentumMatrixDotTimesV(cache)
        h_com = np.dot(A_com, qd)[np.newaxis].T
        h_com_nom = np.vstack([np.zeros((3,1)),139*xd_com_nom])  # desired angular velocity is zero,
                                                                    # CoM velocity matches the CoM trajectory
        hd_com_des = Kp_h*(h_com_nom - h_com)

        # Computed desired accelerations of the feet (at the corner points)
        xdd_left_des, xdd_right_des = self.get_desired_foot_accelerations(cache, 
                                                                          time, 
                                                                          qd, 
                                                                          Kp_foot, Kd_foot)
        # Specify support phase
        support = self.fsm.SupportPhase(time)
       
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

        contact_jacobians, contact_jacobians_dot_v = self.get_contact_jacobians(cache, support)
        num_contacts = len(contact_jacobians)
        
        ################## QP Formulation #################
        
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

        # Foot tracking costs: add a cost for each corner of the foot
        corners = self.get_foot_contact_points()
        for i in range(len(corners)):
            # Left foot tracking cost for this corner
            J_left, Jd_qd_left = self.get_body_jacobian(cache, self.left_foot_index, relative_position=corners[i])
            self.AddJacobianTypeCost(J_left, qdd, Jd_qd_left, xdd_left_des[i], weight=w4)

            # Right foot tracking cost
            J_right, Jd_qd_right = self.get_body_jacobian(cache, self.right_foot_index, relative_position=corners[i])
            self.AddJacobianTypeCost(J_right, qdd, Jd_qd_right, xdd_right_des[i], weight=w4)
            
        # Contact acceleration constraint
        for j in range(num_contacts):
            J_cont = contact_jacobians[j]
            Jd_qd_cont = contact_jacobians_dot_v[j]
            xdd_cont_des = 0

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
        assert result.is_success(), "Whole-body QP Infeasible!"
        return result.GetSolution(tau)

    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """
        q, qd = self.StateToQQDot(state)

        # Compute kinimatics, which will allow us to calculate key quantities
        cache = self.tree.doKinematics(q,qd)

        # Compute the current template and task-space states
        x_lip =  self.x_lip
        x_task = self.GetTaskSpaceState(cache, q, qd)

        print(np.dot(self.K, x_task))
        print("")

        # Generate a template trajectory that respects whole-body CWC constraints
        u_lip, u_task = self.DoTemplateMPC(x_lip, x_task)

        # Feedback linearize with a whole-body QP to get desired torques
        tau = self.SolveWholeBodyQP(cache, context.get_time(), q, qd, u_task)

        # Simulate the template forward in time (simple forward Euler)
        self.x_lip += (np.dot(self.A_lip,self.x_lip) + np.dot(self.B_lip, u_lip))*self.dt

        output[:] = tau

