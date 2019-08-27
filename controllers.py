##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import numpy as np
import time
from pydrake.all import *
from helpers import *
from walking_pattern_generator import *

class AtlasPDController(VectorSystem):
    """
    A simple PD controller that regulates the robot to a nominal (standing) position.
    """
    def __init__(self, tree, plant, Kp=1000.0, Kd=5.0):
        
        # Try to ensure that the MultiBodyPlant that we use for simulation matches the
        # RigidBodyTree that we use for computations
        assert tree.get_num_actuators() == plant.num_actuators()
        assert tree.get_num_velocities() == plant.num_velocities()

        VectorSystem.__init__(self, 
                              plant.num_positions() + plant.num_velocities(),   # input size [q,qd]
                              plant.num_actuators())   # output size [tau]

        self.np = tree.get_num_positions()
        self.nv = tree.get_num_velocities()
        self.nu = tree.get_num_actuators()

        self.nominal_state = RPYAtlasFixedPointState()  # joint angles and torques for nominal position

        self.tree = tree    # RigidBodyTree describing this robot
        self.plant = plant  # MultiBodyPlant describing this robot

        self.joint_angle_map = MBP_RBT_joint_angle_map(plant,tree)  # mapping from MultiBodyPlant joint angles
                                                                    # to RigidBodyTree joint angles

        self.Kp = Kp       # PD gains
        self.Kd = Kd

    def StateToQQDot(self, state):
        """
        Given a full state vector [q, v] of the MultiBodyPlant model, 
        return the joint space states q and qd for the RigidBodyTree model,
        which assumes a floating base with roll/pitch/yaw coordinates instead of
        quaternion. 
        """
        # Get state of MultiBodyPlant model
        q_MBP = state[:self.np+1]  # [ floatbase_quaternion, floatbase_position, joint_angles ]
        v_MBP = state[self.np+1:]  # [ floatbase_rpy_vel, floatbase_xyz_vel, joint_velocities

        # Extract relevant quantities
        floatbase_quaternion = q_MBP[0:4]
        floatbase_quaternion /= np.linalg.norm(floatbase_quaternion)  # normalize
        floatbase_position = q_MBP[4:7]
        joint_angles_mbp = q_MBP[7:]

        floatbase_rpy_vel = v_MBP[0:3]
        floatbase_xyz_vel = v_MBP[3:6]
        joint_velocities_mbp = v_MBP[6:]

        # Translate quaternions to eulers, joint angles to RigidBodyTree order
        floatbase_euler = RollPitchYaw(Quaternion(floatbase_quaternion)).vector()
        joint_angles_rbt = np.dot(self.joint_angle_map, joint_angles_mbp) 
        joint_velocities_rbt = np.dot(self.joint_angle_map, joint_velocities_mbp)

        # Formulate q,qd in RigidBodyTree formate
        q = np.hstack([floatbase_position, floatbase_euler, joint_angles_rbt])
        qd = np.hstack([floatbase_xyz_vel, floatbase_rpy_vel, joint_velocities_rbt])

        return (q,qd)

    def ComputePDControl(self, q, qd):
        """
        Map state [q,qd] to control inputs [u].
        """
        q_nom = self.nominal_state[0:self.np]
        qd_nom = self.nominal_state[self.np:]

        tau = self.Kp*(q_nom-q) + self.Kd*(qd_nom - qd)

        # Convert torques to actuator space
        B = self.tree.B
        u = np.dot(B.T,tau)

        return u

    def DoCalcVectorOutput(self, context, state, unused, output):
        q,qd = self.StateToQQDot(state)
        u = self.ComputePDControl(q,qd)

        output[:] = u
        

class AtlasQPController(AtlasPDController):
    def __init__(self, tree, plant):
        AtlasPDController.__init__(self, tree, plant)

        #self.fsm = WalkingFSM(n_steps=3,         # Finite State Machine describing CoM trajectory,
        #                      step_length=0.60,   # swing foot trajectories, and stance phases.
        #                      step_height=0.10,
        #                      step_time=0.9)
        self.fsm = StandingFSM()

        self.mu = 0.2             # assumed friction coefficient

        self.mp = MathematicalProgram()  # QP that we'll use for whole-body control

        self.right_foot_index = self.tree.FindBody('r_foot').get_body_index()
        self.left_foot_index = self.tree.FindBody('l_foot').get_body_index()
        self.world_index = self.tree.world().get_body_index()

    def get_foot_contact_points(self):
        """
        Return a tuple of points in the foot frame that represent contact locations. 
        (this is a rough guess based on self.tree.getTerrainContactPoints)
        """
        corner_contacts = (
                            np.array([-0.069, 0.08, -0.09]),
                            np.array([-0.069,-0.08, -0.09]),
                            np.array([ 0.201,-0.08, -0.09]),
                            np.array([ 0.201, 0.08, -0.09])
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
    
    def get_desired_foot_accelerations(self, cache, time, qd, Kp, Kd):
        """
        Compute desired accelerations of the four corners of the given foot, such that
        the foot should remain flat and track the desired center trajectory.
        """

        xdd_left_des = []  # return a list of desired accelerations for each corner of the foot
        xdd_right_des = []

        for corner_point in self.get_foot_contact_points():
            # Left foot
            J_left, Jd_qd_left = self.get_body_jacobian(cache, self.left_foot_index, relative_position=corner_point)
            x_left = self.tree.transformPoints(cache, corner_point, self.left_foot_index, self.world_index)
            xd_left = np.dot(J_left, qd)[np.newaxis].T

            x_left_nom, xd_left_nom = self.fsm.LeftFootTrajectory(time)
            x_left_nom += corner_point[np.newaxis].T

            xdd_left_des.append( Kp*(x_left_nom-x_left) + Kd*(xd_left_nom - xd_left) )
            
            # Right foot
            J_right, Jd_qd_right = self.get_body_jacobian(cache, self.right_foot_index, relative_position=corner_point)
            x_right = self.tree.transformPoints(cache, corner_point, self.right_foot_index, self.world_index)
            xd_right = np.dot(J_right, qd)[np.newaxis].T

            x_right_nom, xd_right_nom = self.fsm.RightFootTrajectory(time)
            x_right_nom += corner_point[np.newaxis].T

            xdd_right_des.append( Kp*(x_right_nom-x_right) + Kd*(xd_right_nom - xd_right) )

        return (xdd_left_des, xdd_right_des)

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

        A_i = np.asarray([[ 1, 1, -self.mu],   # pyramid approximation of CWC for one
                          [-1, 1, -self.mu],   # contact force f \in R^3
                          [-1,-1, -self.mu],
                          [ 1,-1, -self.mu]])

        # We'll formulate as lb <= Ax <= ub, where x=[f_1',f_2',...]'
        A = np.kron(np.eye(num_contacts),A_i)

        ub = np.zeros((4*num_contacts,1))
        lb = -np.inf*np.ones((4*num_contacts,1))

        x = np.vstack([f_contact[i] for i in range(num_contacts)])

        return self.mp.AddLinearConstraint(A=A,lb=lb,ub=ub,vars=x)




    def SolveWholeBodyQP(self, cache, xdd_com_des, hd_com_des, xdd_left_des, xdd_right_des, qdd_des, support="double"):
        """
        Formulates a quadratic program which attempts to regulate the joints to the desired
        accelerations and center of mass to the desired position as follows:
        minimize:
            w_1* || J_com*qdd + Jd_com*qd - xdd_com_des ||^2 + 
            w_2* || A*qdd + Ad*qd - hd_com_des ||^2 +
            w_3* || qdd_des - qdd ||^2 +
            w_4* || J_left*qdd + Jd_left*qd - xdd_left_des ||^2 +
            w_4* || J_right*qdd + Jd_right*qd - xdd_right_des ||^2 +
        subject to:
            
                H*qdd + C = B*tau + sum(J'*f)
                f \in friction cones
                J_cj*qdd + J'_cj*qd == nu
                nu_min <= nu <= nu_max
        Parameters:
            cache         : kinematics cache for computing dynamic quantities
            xdd_com_des   : desired center of mass acceleration
            hd_com_des    : desired centroidal momentum dot
            xdd_left_des  : desired acceleration of the left foot
            xdd_right_des : desired acceleration of the right foot
            qdd_des       : desired joint acceleration
            support       : what stance phase we're in. "double", "left", or "right"
        """

        self.mp = MathematicalProgram()

        
        ############## Tuneable Paramters ################

        w1 = 50.0   # center-of-mass tracking weight
        w2 = 0.02  # centroid momentum weight
        w3 = 0.5   # joint tracking weight
        w4 = 5.0    # foot tracking weight

        nu_min = -0.001   # slack for contact constraint
        nu_max = 0.001
        
        ##################################################

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

        # Solve the whole-body QP
        result = Solve(self.mp)

        assert result.is_success()
        return result.GetSolution(tau)


    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        ############## Tuneable Paramters ################

        Kp_q = 10     # Joint angle PD gains
        Kd_q = 100

        Kp_com = 500   # Center of mass PD gains
        Kd_com = 50

        Kp_h = 10.0    # Centroid momentum P gain

        Kp_foot = 100.0   # foot position PD gains
        Kd_foot = 100.0 

        ##################################################

        q, qd = self.StateToQQDot(state)

        # Run kinematics, which will allow us to calculate key quantities
        cache = self.tree.doKinematics(q, qd)

        # Compute desired joint acclerations
        q_nom = self.nominal_state[0:self.np]
        qd_nom = self.nominal_state[self.np:]
        qdd_des = Kp_q*(q_nom-q) + Kd_q*(qd_nom-qd)
        qdd_des = qdd_des[np.newaxis].T

        # Compute desired center of mass acceleration
        x_com = self.tree.centerOfMass(cache)[np.newaxis].T
        xd_com = np.dot(self.tree.centerOfMassJacobian(cache), qd)[np.newaxis].T
        x_com_nom, xd_com_nom, xdd_com_nom = self.fsm.ComTrajectory(context.get_time())
       
        xdd_com_des = xdd_com_nom + Kp_com*(x_com_nom-x_com) + Kd_com*(xd_com_nom-xd_com)

        print(x_com_nom-x_com)

        # Compute desired centroid momentum dot
        A_com = self.tree.centroidalMomentumMatrix(cache)
        Ad_com_qd = self.tree.centroidalMomentumMatrixDotTimesV(cache)
        h_com = np.dot(A_com, qd)[np.newaxis].T
        h_com_nom = np.zeros((6,1))
        hd_com_des = Kp_h*(h_com_nom - h_com)

        # Computed desired accelerations of the feet (at the corner points)
        xdd_left_des, xdd_right_des = self.get_desired_foot_accelerations(cache, 
                                                                          context.get_time(), 
                                                                          qd, 
                                                                          Kp_foot, Kd_foot)

        # Specify support phase
        support = self.fsm.SupportPhase(context.get_time())


        # Solve the whole-body QP to get desired torques
        u = self.SolveWholeBodyQP(cache, xdd_com_des, hd_com_des, xdd_left_des, xdd_right_des, qdd_des, support)

        output[:] = u
