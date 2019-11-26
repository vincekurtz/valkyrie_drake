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
        self.fsm = WalkingFSM(n_steps=4,
                              step_length=0.5,
                              step_height=0.05,
                              step_time=1.0)
        #self.fsm = StandingFSM()

        # Parameters
        h = 0.967
        g = 9.81
        omega = np.sqrt(g/h)
        m = get_total_mass(tree)
        self.m = m
        self.mu = 0.5

        self.l_max = 0.58*m   # maximum linear momentum of the CoM for CWC linearization

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
        
        Q_task = 10*np.eye(9)
        R_task = 1*np.eye(6)
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

        # Remove redundant rows
        #A_cwc, b_cwc = regularize_linear_constraint(A_cwc,b_cwc)  # Warning: this leads to a significant slowdown!

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
        # Prediction horizon and sampling time
        N = 20
        dt = 0.2
            
        # MPC Parameters
        R_mpc = 10*np.eye(2)      # ZMP tracking penalty
        Q_mpc = 1*np.eye(2)          # CoM velocity penalty
        Qf_mpc = 10*np.eye(2)      # Final CoM velocity penalty

        # Set up a Drake MathematicalProgram
        mp = MathematicalProgram()

        # Create optimization variables
        x_lip = mp.NewContinuousVariables(4,N,"x_lip")    # position and velocity of CoM in plane
        u_lip = mp.NewContinuousVariables(2,N-1,"u_lip")  # center of pressure position on the ground plane

        x_task = mp.NewContinuousVariables(9,N,"x_task")    # CoM position and centroidal momentum
        u_task = mp.NewContinuousVariables(6,N-1,"u_task")  # Spatial force on CoM (centroidal momentum dot)
        
        # Initial condition constraints
        init_task_constraint = mp.AddLinearEqualityConstraint(np.eye(9), x_task_init, x_task[:,0])
        init_lip_constraint  = mp.AddLinearEqualityConstraint(np.eye(4), x_lip_init, x_lip[:,0])

        for i in range(N-1):
            # Add Running Costs
            zmp_des = self.fsm.zmp_trajectory.value(t+dt*i)
            mp.AddQuadraticErrorCost(R_mpc,zmp_des,u_lip[:,i])            # regulate ZMP to track nominal
            mp.AddQuadraticErrorCost(Q_mpc,np.zeros((2,1)),x_lip[2:4,i])  # penalize CoM velocity

            # Add dynamic constraints for the LIPM
            AddForwardEulerDynamicsConstraint(mp, self.A_lip, self.B_lip, 
                                                  x_lip[:,i], u_lip[:,i], x_lip[:,i+1],
                                                  dt)

            # Add dynamic constraints for the task space
            AddForwardEulerDynamicsConstraint(mp, self.A_task, self.B_task, 
                                                  x_task[:,i], u_task[:,i], x_task[:,i+1],
                                                  dt)

	    ## Add (exact) interface constraint
            #A_interface = np.hstack([self.R, (self.Q-np.dot(self.K,self.P)), self.K, -np.eye(6)])
            #x_interface = np.hstack([u_lip[:,i],x_lip[:,i],x_task[:,i],u_task[:,i]])[np.newaxis].T
            #interface_con = mp.AddLinearEqualityConstraint(A_interface, np.zeros((6,1)), x_interface)

            if i >= 1:
                # Add (approximate) interface constraint
                
                # Reformulate constraint V(x_task, x_lip) <= epsilon as a quadratic constraint
                # xbar'*QQ*xbar + bb'*xbar + cc <= 0
                epsilon = 5000
                xbar = np.vstack([x_task[:,i][np.newaxis].T,x_lip[:,i][np.newaxis].T])

                QQ = np.vstack( [ np.hstack([ self.M,                   -np.dot(self.M,self.P) ]),
                                  np.hstack([ -np.dot(self.P.T,self.M), np.dot(np.dot(self.P.T,self.M),self.P) ])
                                  ])
                
                bb = np.zeros(xbar.shape)
                cc = -epsilon^2

                # Slight diagonal inflation? This seems hacky, but the few non-positive eigenvalues are very
                # close to zero
                QQ = QQ + 1e-9*np.eye(13)
                AddQuadraticConstraint(mp,QQ,bb,cc,xbar)

                # Get linearized contact constraints
                A_cwc, b_cwc = self.ComputeLinearizedContactConstraint(t+i*dt)
                A_bnd, b_bnd = self.ComputeAccelerationBoundConstraint()

                # Add contact wrench cone constraint
                xbar_cwc = np.hstack([x_task[:,i],u_task[:,i]])[np.newaxis].T  # [x_task;u_task]
                lb_cwc = -np.inf*np.ones(b_cwc.shape) 
                ub_cwc = b_cwc                        
                mp.AddLinearConstraint(A_cwc, lb_cwc, ub_cwc, xbar_cwc)

                ## Add acceleration bound constraint
                lb_bnd = -np.inf*np.ones(b_bnd.shape)
                ub_bnd = b_bnd
                mp.AddLinearConstraint(A_bnd, lb_bnd, ub_bnd, u_task[:,i])

        # Add terminal cost
        mp.AddQuadraticErrorCost(Qf_mpc,np.zeros((2,1)),x_lip[2:4,N-1])

        # Solve the QP
        #solver = OsqpSolver()
        solver = GurobiSolver()
        #solver = IpoptSolver()
        res = solver.Solve(mp,None,None)

        assert res.is_success(), "Template MPC Failed"
        if not res.is_success():
            print("Template MPC Failed!")

        u_lip_trajectory = res.GetSolution(u_lip)
        u_task_trajectory = res.GetSolution(u_task)

        return u_lip_trajectory, u_task_trajectory

    def SolveWholeBodyQP(self, cache, context, q, qd, u_task):
        """
        Formulates and solves a quadratic program which attempts to regulate the joints to the desired
        accelerations and center of mass to the desired position as follows:

        minimize:
            w1* || A*qdd + Ad*qd - u_task ||^2 +
            w2* || qdd_des - qdd ||^2 +
            w3* || J_foot*qdd + Jd_foot*qd - xdd_foot_des ||^2 +
            w4* || J_torso*qdd + Jd_torso*qd - rpydd_torso_des ||^2
        subject to:
                H*qdd + C = B*tau + sum(J'*f)
                f \in friction cones
                J_cj*qdd + J'_cj*qd == nu
                nu_min <= nu <= nu_max
        """

        ############## Tuneable Paramters ################

        w1 = 1.0    # centroid momentum weight
        w2 = 10.0    # joint tracking weight
        w3 = 200.0   # foot tracking weight
        w4 = 100.0   # torso orientation weight

        nu_min = -1e-10   # slack for contact constraint
        nu_max = 1e-10

        Kp_q = 100     # Joint angle PD gains
        Kd_q = 10

        Kp_foot = 100.0   # foot position PD gains
        Kd_foot = 10.0 

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

        A = self.tree.centroidalMomentumMatrix(cache)              # Centroidal "jacobians"
        Ad_qd = self.tree.centroidalMomentumMatrixDotTimesV(cache)

        J_left, Jd_qd_left = self.get_body_jacobian(cache, self.left_foot_index)  # foot jacobians
        J_right, Jd_qd_right = self.get_body_jacobian(cache, self.right_foot_index)

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

        # Computed desired accelerations of the feet (at the corner points)
        xdd_left_des, xdd_right_des = self.get_desired_foot_accelerations(cache, 
                                                                          context.get_time(), 
                                                                          qd, 
                                                                          Kp_foot, Kd_foot)

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

        # Centroidal momentum cost
        centroidal_cost = self.AddJacobianTypeCost(A, qdd, Ad_qd, u_task[:,0], weight=w1)
       
        # Joint tracking cost
        joint_cost = self.mp.AddQuadraticErrorCost(Q=w2*np.eye(self.nv),x_desired=qdd_des,vars=qdd)

        # Foot tracking costs: add a cost for each corner of the foot
        corners = self.get_foot_contact_points()
        for i in range(len(corners)):
            # Left foot tracking cost for this corner
            J_left, Jd_qd_left = self.get_body_jacobian(cache, self.left_foot_index, relative_position=corners[i])
            self.AddJacobianTypeCost(J_left, qdd, Jd_qd_left, xdd_left_des[i], weight=w3)

            # Right foot tracking cost
            J_right, Jd_qd_right = self.get_body_jacobian(cache, self.right_foot_index, relative_position=corners[i])
            self.AddJacobianTypeCost(J_right, qdd, Jd_qd_right, xdd_right_des[i], weight=w3)
        
        # torso orientation cost
        torso_cost = self.AddJacobianTypeCost(J_torso, qdd, Jd_qd_torso, rpydd_torso_des, weight=w4)
            
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

        assert result.is_success(), "Whole-body QP Failed"

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

        # Generate a template trajectory that respects whole-body CWC constraints
        u_lip_traj, u_task_traj = self.DoTemplateMPC(context.get_time(), x_lip, x_task)
        u_lip = u_lip_traj[:,0][np.newaxis].T
        u_task = u_task_traj[:,0][np.newaxis].T

        # Feedback linearize with a whole-body QP to get desired torques
        tau = self.SolveWholeBodyQP(cache, context, q, qd, u_task)#, x_com_nom, xd_com_nom)

        # Simulate the template forward in time (simple forward Euler)
        self.x_lip += (np.dot(self.A_lip,self.x_lip) + np.dot(self.B_lip, u_lip))*self.dt

        output[:] = tau
