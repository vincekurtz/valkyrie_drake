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
        #self.fsm = WalkingFSM(n_steps=3,
        #                      step_length=0.6,
        #                      step_height=0.1,
        #                      step_time=0.9)
        self.fsm = StandingFSM()

        # Parameters
        h = 0.967
        g = 9.81
        omega = np.sqrt(g/h)
        m = get_total_mass(tree)

        self.l_max = 100.0   # maximum linear momentum of the CoM for CWC linearization

        # Wrench on the CoM due to gravity
        self.w_mg = np.array([0,0,0,0,0,-m*g])[np.newaxis].T

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
        
        Q_task = np.eye(9)
        R_task = np.eye(6)
        K, P = LinearQuadraticRegulator(self.A_task, self.B_task, Q_task, R_task)
        self.K = -K

        lmbda = 0.001
        M = interface_mp.NewSymmetricContinuousVariables(9,"M")

        interface_mp.AddPositiveSemidefiniteConstraint(M - np.dot(self.C_task.T, self.C_task))

        AplusBK = self.A_task + np.dot(self.B_task, self.K)
        interface_mp.AddPositiveSemidefiniteConstraint(-2*lmbda*M - np.dot(AplusBK.T,M) - np.dot(M,AplusBK) )

        result = Solve(interface_mp)

        assert result.is_success(), "Interface SDP infeasible"
        self.M = result.GetSolution(M)

        self.P = self.C_lip

        self.Q = np.zeros((6,4))
        self.Q[3:5,0:2] = omega**2*m*np.eye(2)

        self.R = np.zeros((6,2))
        self.R[3:5] = -m*omega**2*np.eye(2)

        # Double check the results
        assert is_pos_def(self.M) , "M is not positive definite."
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

    def comForceTransform(self, cache):
        """
        Returns the 6x6 matrix O_Xf_com that transforms
        forces in the center of mass frame to the world frame
        """
        p_com = self.tree.centerOfMass(cache).flatten()
        O_Xf_com = np.eye(6)
        O_Xf_com[0:3,3:6] = S(p_com)

        return O_Xf_com

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

    def ComputeGIWC(self, t):
        """
        Compute the Gravito Intertial Wrench Cone for the current stance

            A f_{com} <= 0,

        where f_{com} is the wrench on the center of mass expressed in the 
        world frame. 
        """

        support = self.fsm.SupportPhase(t)
        
        # Friction pyramid for forces at each vertex of each contact surface
        # A_vertex*f_vertex <= 0 ensures Coulomb friction is obeyed
        A_vertex = np.array([[ 1.0, 0.0, -self.mu],
                             [-1.0, 0.0, -self.mu],
                             [ 0.0, 1.0, -self.mu],
                             [ 0.0,-1.0, -self.mu]])

        # Net friction cone for forces f_all = [f_vertex_1, f_vertex_2,...]' 
        # at four vertices of each contact surface.
        # A_all*f_all <= 0 ensures Coulomb friction is obeyed
        A_all = np.kron(np.eye(4),A_vertex)

        # Surface wrench w_surf = [tau_surf, f_surf]' resulting from f_all
        # w_surf = G_surf*f_all
        G_surf = np.zeros((6,12))

        G_surf[3:6,:] = np.hstack([np.eye(3),np.eye(3),np.eye(3),np.eye(3)])  # sum of forces

        p1,p2,p3,p4 = self.get_foot_contact_points() # sum of torques
        G_surf[0:3,0:3] = S(p1)                        
        G_surf[0:3,3:6] = S(p4)
        G_surf[0:3,6:9] = S(p2)
        G_surf[0:3,9:12] = S(p3)

        # Get foot (contact) positions in the world frame
        rf_pos, rf_vel = self.fsm.RightFootTrajectory(t)
        lf_pos, lf_vel = self.fsm.LeftFootTrajectory(t)
        
        # Centroidal wrench w_GI = G_stance*w_all, where w_all = [w_surf_1, w_surf_2, ...]'
        # G_stance = [ R, S(p)*R ]
        #            [ 0,   R    ]
        if support == "right":
            # We'll assume contacts are not rotated in the world frame, so R = eye(3)
            G_stance = np.eye(6)
            G_stance[0:3,3:6] = S(rf_pos.flatten())
        elif support == "left":
            G_stance = np.eye(6)
            G_stance[0:3,3:6] = S(lf_pos.flatten())
        elif support == "double":
            # w_GI = G_stance_right*w_right + G_stance_left*w_left
            G_stance = np.zeros((6,12))
            G_stance[0:6,0:6] = np.eye(6)
            G_stance[0:3,3:6] = S(rf_pos.flatten())
            G_stance[0:6,6:12] = np.eye(6)
            G_stance[0:3,9:12] = S(lf_pos.flatten())
        else:
            raise ValueError("Unknown support mode '%s'." % support)

        # Use cone double description to get vertex constraints in span form
        V_all = face_to_span(A_all)

        # Compute surfance wrench cone in span form
        V_surf = np.dot(G_surf, V_all)

        # Compute centroidal wrench cone in span form
        if support == "right" or support == "left":
            V_centroid = np.dot(G_stance, V_surf)
        elif support == "double":
            V_centroid = np.dot(G_stance, np.vstack([V_surf,V_surf]))

        # Convert centroidal wrench cone to face form
        A_centroid = span_to_face(V_centroid)

        return A_centroid

    def ComputeLinearizedContactConstraint(self, t):
        """
        Compute the linearized contact constraint

            A_cwc*[x_task] <= b_{cwc}
                  [u_task]

        for the current stance, based on linearizing the bilinear GIWC constraint
        based on bounding the CoM acceleration.
        """
        # Centroidal wrench cone expressed in the world frame
        # such that A*0_Xf_com*(u_com-mg) <= 0 enforces the CWC constraint
        A = self.ComputeGIWC(t)

        A_cwc = np.zeros((8*A.shape[0],15))
        b_cwc = np.zeros((8*A.shape[0],1))

        # Iterate over corners of the cube constrainting CoM accelerations
        unit_corners = list(itertools.product(*zip([-1,-1,-1],[1,1,1])))
        for i in range(8):
            unit_corner = unit_corners[i]
            l_dot = self.l_max*np.asarray(unit_corner)

            # Compute the constraint 
            #   A*u_com + A*[S(mg)-S(l_dot)]*p_com <= A*[0 ]
            #               [    0         ]            [mg]
            # for this value of l_dot
            mg = self.w_mg[3:6,0]
            cross_term = np.vstack([ S(mg)-S(l_dot),
                                     np.zeros((3,3))]) 

            A_cwc_i = np.hstack( [ np.dot(A,cross_term), np.zeros((16,6)), A ])
            b_cwc_i = np.dot(A,self.w_mg)
          
            # Add this constraint to the stack of all constraints
            start_idx = i*A.shape[0]
            end_idx = (i+1)*A.shape[0]
            A_cwc[start_idx:end_idx,:] = A_cwc_i

            b_cwc[start_idx:end_idx,:] = b_cwc_i

        return A_cwc, b_cwc

    def ComputeAccelerationBoundConstraint(self):
        """
        Compute the constraint

            A_bnd*u_task <= b_bound

        which enforces that the CoM acceleration is bounded
        by self.l_max in the sense that

            \| m pdd_com \|_infty <= l_max
        """
        A_bnd = np.array([[0,0,0, 1, 0, 0],
                          [0,0,0,-1, 0, 0],
                          [0,0,0, 0, 1, 0],
                          [0,0,0, 0,-1, 0],
                          [0,0,0, 0, 0, 1],
                          [0,0,0, 0, 0,-1]])

        b_bnd = self.l_max*np.ones((6,1))

        return A_bnd, b_bnd

    def DoTemplateMPC(self, t, x_lip_init, x_task_init):
        """
        Given the current template (x_lip) and task space (x_task) states, perform MPC
        in the template model while respecting contact constraints for the full model.
        """
        # Get linearized contact constraints
        A_cwc, b_cwc = self.ComputeLinearizedContactConstraint(t)
        A_bnd, b_bnd = self.ComputeAccelerationBoundConstraint()

        # Specify Parameters
        Q_mpc = 10*np.eye(4)
        Qf_mpc = np.eye(4)
        R_mpc = np.eye(2)
        N = 10

        # Set up a Drake MathematicalProgram
        mp = MathematicalProgram()

        # Create optimization variables
        x_lip = mp.NewContinuousVariables(4,N,"x_lip")
        u_lip = mp.NewContinuousVariables(2,N-1,"u_lip")
        x_task = mp.NewContinuousVariables(9,N,"x_task")
        u_task = mp.NewContinuousVariables(6,N-1,"u_task")

        # Initial condition constraints
        mp.AddLinearEqualityConstraint(np.eye(4), x_lip_init, x_lip[:,0])
        mp.AddLinearEqualityConstraint(np.eye(9), x_task_init, x_task[:,0])

        for i in range(N-1):
            # Add Running Costs
            x_lip_des = np.zeros(x_lip_init.shape)
            u_lip_des = np.zeros((2,1))
            mp.AddQuadraticErrorCost(Q_mpc,x_lip_des,x_lip[:,i])
            mp.AddQuadraticErrorCost(R_mpc,u_lip_des,u_lip[:,i])

            # Add dynamic constraints for the LIPM
            A_bar_lip = np.hstack([self.A_lip, self.B_lip, -np.eye(4)])
            x_bar_lip = np.hstack([x_lip[:,i],u_lip[:,i],x_lip[:,i+1]])[np.newaxis].T
            mp.AddLinearEqualityConstraint(A_bar_lip, np.zeros((4,1)),x_bar_lip)

            # Add dynamic constraints for the task space
            A_bar_task = np.hstack([self.A_task, self.B_task, -np.eye(9)])
            x_bar_task = np.hstack([x_task[:,i],u_task[:,i],x_task[:,i+1]])[np.newaxis].T
            mp.AddLinearEqualityConstraint(A_bar_task, np.zeros((9,1)),x_bar_task)

            # Add interface constraint
            A_interface = np.hstack([self.R, (self.Q-np.dot(self.K,self.P)), self.K, -np.eye(6)])
            x_interface = np.hstack([u_lip[:,i],x_lip[:,i],x_task[:,i],u_task[:,i]])[np.newaxis].T
            mp.AddLinearEqualityConstraint(A_interface, np.zeros((6,1)), x_interface)

            # Add contact wrench cone constraint
            x_u = np.hstack([x_task[:,i],u_task[:,i]])[np.newaxis].T  # [x_task;u_task]
            lb = -np.inf*np.ones(b_cwc.shape) # lower bound
            ub = b_cwc                        # upper bound
            mp.AddLinearConstraint(A_cwc, lb, ub, x_u)

        # Add terminal cost
        x_lip_des = np.zeros(x_lip_init.shape)
        mp.AddQuadraticErrorCost(Qf_mpc,x_lip_des,x_lip[:,N-1])
            
        # Solve the QP
        solver = OsqpSolver()
        res = solver.Solve(mp,None,None)

        u_lip_trajectory = res.GetSolution(u_lip)
        u_task_trajectory = res.GetSolution(u_task)

        return u_lip_trajectory, u_task_trajectory

    def CheckContactFrictionConstraint(self, f_contact):
        """
        Given a set of contact forces, check to see if they obey Coulomb Frction
        """
        num_contacts = len(f_contact)

        A_i = np.asarray([[ 1, 0, -self.mu],  # Pyramid approximation of CWC for one 
                          [-1, 0, -self.mu],  # contact force f \in R^3
                          [ 0, 1, -self.mu],
                          [ 0,-1, -self.mu]])

        # We'll formulate the check as Ax <= 0, where x = [f_1; f_2; ...]
        A = np.kron(np.eye(num_contacts),A_i)
        x = np.vstack(f_contact)

        return np.all(np.dot(A,x) <= 0)

    def CorrectContactFriction(self, J_task, H, B, contact_jacobians, tau_nom, f_contact_nom):
        """
        Given nominal torques and contact forces that do NOT obey Coulomb friction, but do
        enforce the task (with jacobian J_task), generate new torques that have the same effect
        on the task but do obey friction. 
        """

        num_contacts = len(f_contact_nom)

        self.mp = MathematicalProgram()

        tau = self.mp.NewContinuousVariables(self.nu,1,'tau')
        f_contact = [self.mp.NewContinuousVariables(3,1,'f_%s'%i) for i in range(num_contacts)]

        # Cost function: we want the corrected torques to be as close as possible to the nominal ones
        self.mp.AddQuadraticErrorCost(Q=1*np.eye(self.nu),x_desired=tau_nom,vars=tau)
        self.mp.AddQuadraticErrorCost(Q=0.001*np.eye(self.nu),x_desired=np.zeros(tau_nom.shape),vars=tau)

        # Constraint. This will ensure that the task space accelerations are the same as those
        # generated with the nominal torques:
        #
        # [J_task*H^{-1}*B  J_task*H^{-1}*J_c']*([tau;f_c] - [tau_nom;f_c_nom]) = 0
        #
        J_contact = np.vstack(contact_jacobians)
        J_task = np.vstack([J_task,J_contact])

        JH = np.dot(J_task,np.linalg.inv(H))
        JHB = np.dot(JH,B)
        Jc = np.hstack([contact_jacobians[i].T for i in range(num_contacts)])
        JHJc = np.dot(JH,Jc)

        A = np.hstack([JHB, JHJc])             # Formulate as Ax = b
        x = np.vstack([tau, np.vstack(f_contact)])
        x_nom = np.vstack([tau_nom[np.newaxis].T, np.vstack(f_contact_nom)])
        b = np.dot(A,x_nom)
        self.mp.AddLinearEqualityConstraint(A,b,x)

        # Add a friction constraint
        self.AddFrictionPyramidConstraint(f_contact)

        # Solve the optimizaiton
        res = Solve(self.mp)
        tau = res.GetSolution(tau)
        f_contact = [res.GetSolution(f_contact[i])[np.newaxis].T for i in range(num_contacts)]

        return tau, f_contact


    def SolveWholeBodyQP(self, cache, context, q, qd, u_task):
        """
        Use a whole-body quadratic program to feedback linearize the task-space
        dynamics.
        """

        ############## Tuneable Paramters ################

        Kp_q = 1     # Joint angle PD gains
        Kd_q = 1

        Kp_foot = 00.0   # foot position PD gains
        Kd_foot = 100.0 
        
        Kd_contact = 100.0  # Contact movement damping P gain

        w1 = 100.1   # joint tracking weight
        w2 = 50.0    # foot tracking weight

        nu_min = -0.001   # slack for contact constraint
        nu_max = 0.001

        ##################################################

        # Compute desired joint acclerations
        q_nom = self.nominal_state[0:self.np]
        qd_nom = self.nominal_state[self.np:]
        qdd_des = Kp_q*(q_nom-q) + Kd_q*(qd_nom-qd)
        qdd_des = qdd_des[np.newaxis].T

        # Computed desired accelerations of the feet (at the corner points)
        xdd_left_des, xdd_right_des = self.get_desired_foot_accelerations(cache, 
                                                                          context.get_time(), 
                                                                          qd, 
                                                                          Kp_foot, Kd_foot)
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

        contact_jacobians, contact_jacobians_dot_v = self.get_contact_jacobians(cache, support)
        num_contacts = len(contact_jacobians)
        
        ################## QP Formulation #################
        
        self.mp = MathematicalProgram()
        
        # create optimization variables
        qdd = self.mp.NewContinuousVariables(self.nv, 1, 'qdd')   # joint accelerations
        tau = self.mp.NewContinuousVariables(self.nu, 1, 'tau')   # applied torques
       
        f_contact = [self.mp.NewContinuousVariables(3,1,'f_%s'%i) for i in range(num_contacts)]

        # Joint tracking cost
        joint_cost = self.mp.AddQuadraticErrorCost(Q=w1*np.eye(self.nv),x_desired=qdd_des,vars=qdd)

        # Foot tracking costs: add a cost for each corner of the foot
        corners = self.get_foot_contact_points()
        for i in range(len(corners)):
            # Left foot tracking cost for this corner
            J_left, Jd_qd_left = self.get_body_jacobian(cache, self.left_foot_index, relative_position=corners[i])
            self.AddJacobianTypeCost(J_left, qdd, Jd_qd_left, xdd_left_des[i], weight=w2)

            # Right foot tracking cost
            J_right, Jd_qd_right = self.get_body_jacobian(cache, self.right_foot_index, relative_position=corners[i])
            self.AddJacobianTypeCost(J_right, qdd, Jd_qd_right, xdd_right_des[i], weight=w2)
            
        # Centroidal momentum constraint
        hd_com_des = u_task[:,0]
        centroidal_constraint = self.AddJacobianTypeConstraint(A, qdd, Ad_qd, hd_com_des)
       
        # Contact acceleration constraint
        #for j in range(num_contacts):
        #    J_cont = contact_jacobians[j]
        #    Jd_qd_cont = contact_jacobians_dot_v[j]
        #    xd_cont = np.dot(J_cont,qd)[np.newaxis].T
        #    xdd_cont_des = -Kd_contact*xd_cont

        #    contact_constraint = self.AddJacobianTypeConstraint(J_cont, qdd, Jd_qd_cont, xdd_cont_des)
 
        #    # add some slight flexibility to this constraint to avoid infeasibility
        #    contact_constraint.evaluator().UpdateUpperBound(nu_max*np.array([1.0,1.0,1.0]))
        #    contact_constraint.evaluator().UpdateLowerBound(nu_min*np.array([1.0,1.0,1.0]))
       
        # Dynamic constraints 
        dynamics_constraint = self.AddDynamicsConstraint(H, qdd, C, B, tau, contact_jacobians, f_contact)

        # DEBUG: incentivize high torques to make ground contact slip. This is not a problem if
        # we have friction constraints, but causes problems if we remove the friction constraints.
        # We can use this sort of idea to debug post-processing
        # steps that allow us to project friction constraints to the template but correct for issues with
        # a kinematic chain using a null-space projector.
        idx1 = 4
        idx2 = 10
        Q_tau = np.zeros((self.nu,self.nu))
        Q_tau[idx1,idx1] = 1.1
        Q_tau[idx2,idx2] = 1.1
        tau_des = np.zeros((self.nu,1))
        tau_des[idx1] = -340
        tau_des[idx2] = 340
        self.mp.AddQuadraticErrorCost(Q=Q_tau,x_desired=tau_des,vars=tau)

        # Friction cone (really pyramid) constraints 
        friction_constraint = self.AddFrictionPyramidConstraint(f_contact)

        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(self.mp,None,None)
        assert result.is_success(), "Whole-body QP Infeasible!"
        tau = result.GetSolution(tau)

        # Post-processing step: check if the torques obey friction
        f_contact = [result.GetSolution(f_i)[np.newaxis].T for f_i in f_contact]

        respects_friction = self.CheckContactFrictionConstraint(f_contact)
        print(respects_friction)
        if not respects_friction:
            # Make a small correction to tau so that it respects friction
            tau, f_contact = self.CorrectContactFriction(A, H, B, contact_jacobians, tau, f_contact)

        return tau

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

        # Generate a template trajectory that respects whole-body CWC constraints
        #u_lip_traj, u_task_traj = self.DoTemplateMPC(context.get_time(), x_lip, x_task)
        #u_task = u_task_traj[:,0][np.newaxis].T

        # DEBUG: for now, this is much faster than performing template MPC
        u_task = np.zeros((6,1))


        # Feedback linearize with a whole-body QP to get desired torques
        tau = self.SolveWholeBodyQP(cache, context, q, qd, u_task)

        # Simulate the template forward in time (simple forward Euler)
        #self.x_lip += (np.dot(self.A_lip,self.x_lip) + np.dot(self.B_lip, u_lip))*self.dt

        output[:] = tau
