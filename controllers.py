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

class WalkingFSM(object):
    """
    A finite state machine describing a simple walking motion. Specifies support phase
    and center of mass position as functions of time. 
    """
    def __init__(self):
        self.t_switch = 1.0

    def SupportPhase(self, time):
        """
        Return the current support phase, "double", "left", or "right".
        """
        if time <= self.t_switch:
            return "double"
        else:
            return "left"

    def ComTrajectory(self, time):
        """
        Return a desired center of mass position and velocity for the given timestep
        """
        # for now we'll consider this trajectory to be a straight line connecting 
        # two points

        if time <= self.t_switch:
            x_com =  [0.0, (0.1/self.t_switch)*time, 1.0]
            xd_com = [0.0, 0.1/self.t_switch, 0.0]
        else:
            x_com = [0.0, 0.1, 1.0]
            xd_com = [0.0, 0.0, 0.0]

        return (x_com, xd_com)

    def SwingFootTrajectory(self, time):
        """
        Specify a desired position and velocity for the swing foot
        """
        # Right now just specify a trajectory for the right foot
        if time <= self.t_switch:
            x_foot = [-0.075,-0.153,0.0827]
            xd_foot = [0.0, 0.0, 0.0]
        else:
            x_foot = [-0.075+0.5*(time-self.t_switch),-0.153,0.0827]
            xd_foot = [0.5, 0.0, 0.0]

        return (x_foot, xd_foot)

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

        self.nominal_state = RPYValkyrieFixedPointState()  # joint angles and torques for nominal position
        self.tau_ff = RPYValkyrieFixedPointTorque()

        self.tree = tree   # rigid body tree describing this robot

        self.Kp = Kp       # PD gains
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
                                      tree,
                                      Kp=10,
                                      Kd=100)

        self.fsm = WalkingFSM()   # Finite State Machine describing CoM trajectory,
                                  # swing foot trajectories, and stance phases.

        self.mu = 0.3   # friction coefficient

        self.mp = MathematicalProgram()  # QP that we'll use for whole-body control

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


    def FormulateWholeBodyQP(self, cache, xdd_com_des, xdd_foot_des, qdd_des, support="double"):
        """
        Solve a quadratic program which attempts to regulate the joints to the desired
        accelerations and center of mass to the desired position as follows:

            min   w_1*| J_com*qdd + J_comd*qd - xdd_com_des |^2 + w_2*| qdd_des - qdd |^2
            s.t.  H*qdd + C = B*tau + sum(J'*f)
                  f \in friction cones
                  J*qdd + Jd*qd == 0  for all contacts

        Parameters:
            cache        : kinematics cache for computing dynamic quantities
            xdd_com_des  : desired center of mass acceleration
            xdd_foot_des : desired acceleration of the swing foot, used if support="left" or "right"
            qdd_des      : desired joint acceleration
            support      : what stance phase we're in. "double", "left", or "right"

        Returns:
            tau          : joint torques in control space, ready to be applied as outputs
        """

        self.mp = MathematicalProgram()

        # Set weightings for prioritized task-space control
        w1 = 1.0   # center-of-mass tracking
        w2 = 0.1   # joint tracking
        w3 = 1.5   # foot tracking

        # Compute dynamic quantities. Note that we cast vectors as nx1 numpy arrays to allow
        # for matrix multiplication with np.dot().
        H = self.tree.massMatrix(cache)                  # Equations of motion
        C = self.tree.dynamicsBiasTerm(cache, {}, True)[np.newaxis].T
        B = self.tree.B

        J_com = self.tree.centerOfMassJacobian(cache)    # Center of mass jacobian
        Jd_qd_com = self.tree.centerOfMassJacobianDotTimesV(cache)[np.newaxis].T
        
        if support == "left":                            # swing foot jacobians
            swing_foot_index = self.tree.FindBody('rightFoot').get_body_index()
            J_foot, Jd_qd_foot = self.get_swing_foot_jacobian(cache,swing_foot_index)
        elif support == "right":
            swing_foot_index = self.tree.FindBody('leftFoot').get_body_index()
            J_foot, Jd_qd_foot = self.get_swing_foot_jacobian(cache,swing_foot_index)

        contact_jacobians = self.get_contact_jacobians(cache, support)
        num_contacts = len(contact_jacobians)

        # Cast desired quantities (for tracking) as nx1 numpy arrays
        xdd_com_des = xdd_com_des[np.newaxis].T
        xdd_foot_des = xdd_foot_des[np.newaxis].T
        qdd_des = qdd_des[np.newaxis].T

        # create optimization variables
        qdd = self.mp.NewContinuousVariables(self.nv, 1, 'qdd')   # joint accelerations
        tau = self.mp.NewContinuousVariables(self.nu, 1, 'tau')   # applied torques
       
        f_contact = [self.mp.NewContinuousVariables(3,1,'f_%s'%i) for i in range(num_contacts)]

        # Center of mass tracking cost
        com_cost = self.AddJacobianTypeCost(J_com, qdd, Jd_qd_com, xdd_com_des, weight=w1)
       
        # Joint tracking cost
        joint_cost = self.mp.AddQuadraticErrorCost(Q=w2*np.eye(self.nv),x_desired=qdd_des,vars=qdd)

        # Foot tracking cost
        if support != "double":
            foot_cost = self.AddJacobianTypeCost(J_foot, qdd, Jd_qd_foot, xdd_foot_des, weight=w3)
       
        # Dynamic constraints 
        dynamics_constraint = self.AddDynamicsConstraint(H, qdd, C, B, tau, contact_jacobians, f_contact)

        # Friction cone (really pyramid) constraints 
        friction_constraint = self.AddFrictionPyramidConstraint(f_contact)

        return tau  # a reference to the symbolic variable we'll use the value of as the applied control


    def DoCalcVectorOutput(self, context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        q, qd = self.StateToQV(state)

        # Run kinematics, which will allow us to calculate key quantities
        cache = self.tree.doKinematics(q,qd)

        # Compute desired joint acclerations
        q_nom, qd_nom = self.StateToQV(self.nominal_state)
        qdd_des = 1*(q_nom-q) + 10*(qd_nom-qd)

        # Compute desired center of mass acceleration
        x_com = self.tree.centerOfMass(cache)
        xd_com = np.dot(self.tree.centerOfMassJacobian(cache), qd)
        x_com_nom, xd_com_nom = self.fsm.ComTrajectory(context.get_time())

        xdd_com_des = 100*(x_com_nom-x_com) + 100*(xd_com_nom-xd_com)

        # Compute desired swing (right) foot acceleration
        swing_foot_index = self.tree.FindBody('rightFoot').get_body_index()
        J_foot, Jd_qd_foot = self.get_swing_foot_jacobian(cache,swing_foot_index)
        x_foot = self.tree.transformPoints(cache,[0,0,0],swing_foot_index,0).flatten()
        xd_foot = np.dot(J_foot,qd)
        x_foot_nom, xd_foot_nom = self.fsm.SwingFootTrajectory(context.get_time())
        
        xdd_foot_des = 100*(x_foot_nom-x_foot) + 1*(xd_foot_nom-xd_foot)

        # Specify support phase
        support = self.fsm.SupportPhase(context.get_time())

        # Formulate the whole-body QP to get desired torques
        tau = self.FormulateWholeBodyQP(cache, xdd_com_des, xdd_foot_des, qdd_des, support)

        # Solve the whole-body QP
        result = Solve(self.mp)

        if result.is_success():
            u = result.GetSolution(tau)
        else:
            print("Whole-body QP Solver Failed! Falling back to PD controller")
            u = self.ComputePDControl(q,qd,feedforward=True)

        output[:] = u
