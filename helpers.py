##
#
# Helper functions for simulating a Valkyrie humanoid. 
#
##
import numpy as np
import time
from pydrake.all import *

def ManipulatorDynamics(plant, q, v=None):
    """
    Return manipulator dynamic quantities M, Cv, tauG, B, and tauExt,
    where

    M(q)v + C(q,v)v = tauG + Bu + tauExt.

    Adopted from http://underactuated.mit.edu/underactuated.html?chapter=intro.
    """
    assert isinstance(plant,MultibodyPlant)

    # Note that in the long term it would probably be worthwhile to create symbolic 
    # functions for these quantities.
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, q)
    if v is not None:
        plant.SetVelocities(context, v)
    M = plant.CalcMassMatrixViaInverseDynamics(context)
    Cv = plant.CalcBiasTerm(context)
    tauG = plant.CalcGravityGeneralizedForces(context)
    B = plant.MakeActuationMatrix()

    # External forces assumed to be zero here
    forces = MultibodyForces(plant)
    plant.CalcForceElementsContribution(context, forces)
    tauExt = forces.generalized_forces()

    return (M, Cv, tauG, B, tauExt)


def solve_joint_accel_QP(M, Cv, tauG, B, tauExt, vd_des):
    """
    Solve a quadratic program which attempts to track the given desired
    joint accelerations vd_des. Specifically, we will return the actuator inputs u
    that solve the following optimization problem:

        min   | vd - vd_des |^2
        s.t.  M*vd + Cv = tauG + B*u + tauExt

    """

    start_time = time.time()
    # Decision variables
    mp = MathematicalProgram()
    vd = mp.NewContinuousVariables(len(vd_des),'vd')
    u = mp.NewContinuousVariables(B.shape[1],'u')

    # Cast everybody into a np.matrix or vertical numpy array so we can use python2.7's matrix
    # multiplication tools effectively. 

    M = np.matrix(M)              # pre-determinined variables
    Cv = Cv[np.newaxis].T
    tauG = tauG[np.newaxis].T
    B = np.matrix(B)
    tauExt = tauExt[np.newaxis].T
    vd_des = vd_des[np.newaxis].T

    vd = vd[np.newaxis].T         # decision variables
    u = u[np.newaxis].T

    Q = np.eye(len(vd_des))       # cost function parameters
    Q = np.matrix(Q)  

    # Add cost function to the QP
    cost = (vd_des.T-vd.T)*Q*(vd_des-vd)  # cost is a 1x1 np.matrix. We need it to be a pydrake.symbolic.Expression
    mp.AddQuadraticCost(cost[0,0])        # to use it in the mp, so we extract the first element here.

    # Add constraints to the QP
    Mvd = M*vd
    Bu = B*u
    for i in range(M.shape[0]):
        mp.AddLinearConstraint(Mvd[i,0] + Cv[i] - tauG[i] - Bu[i,0] - tauExt[i] == 0)

    # solve the QP
    result = Solve(mp)

    print(time.time() - start_time)

    return result.GetSolution(u)
    


def QP_example():
    """
    Example of using Drake to solve a quadratic program

        min   1/2x'Qx + c'x
        s.t.  Ax <= b
    """
    # Cost and constraint parameters
    Q = np.matrix([[1, -1],     # we need to use matrices and the '*' operator for
                  [-1, 2]])     # matrix multiplication so that we can multiply these mp.Variable objects,
    c = np.matrix([[-2],[-6]])  # which are numpy arrays with dtype=object, which np.matmul doesn't support.

    A = np.matrix([[1,  1],
                  [-1, 2],
                  [2,  1]])

    b = np.matrix([[2],[2],[3]])

    # Set up the problem
    mp = MathematicalProgram()
    x = mp.NewContinuousVariables(2,"x")
    x = x[np.newaxis].T         # formulate as proper (vertical) vector

    # Add the cost
    mp.AddQuadraticCost(Q,c,x)

    # Add constraints elementwise, since drake's (in)equality constraints don't seem to be able
    # to handle vector inequalities.
    Ax = A*x
    for i in range(b.shape[0]):
        Ax_i = Ax[i,0]
        mp.AddLinearConstraint(Ax[i,0] <= b[i])
  
    # Get the solution
    result = Solve(mp)
    print("Result: x = %s" % result.GetSolution(x))
    print("")
    print("Used solver [%s]" % result.get_solver_id().name())
    print("Run time %s s" % result.get_solver_details().run_time)

    return result



def list_joints_and_actuators(robot):
    """
    Run through all the joints and actuators in the robot model 
    and print their names and indeces.
    """
    for i in range(robot.num_actuators()):
        joint_actuator = robot.get_joint_actuator(JointActuatorIndex(i))
        joint = joint_actuator.joint()

        print("Actuator [%s] acts on joint [%s]" % (joint_actuator.name(),joint.name()))


def ValkyrieFixedPointState():
    """
    Return a reasonable initial state for the Valkyrie humanoid, where
    the orientation is expressed in quaternions.
    """

    q = np.zeros(37)
    q[0:4] = [1, 0, 0, 0]   # floating base orientation
    q[4:7] = [0, 0, 1.025]  # floating base position
    q[7] = 0                # spine
    q[8] = 0                # r hip
    q[9] = 0                # l hip
    q[10] = 0               # spine
    q[11] = 0               # r hip
    q[12] = 0               # l hip
    q[13] = 0               # spine
    q[14] = -0.49           # r hip
    q[15] = -0.49           # l hip
    q[16] = 0               # neck
    q[17] = 0.3             # r shoulder
    q[18] = 0.3             # l shoulder
    q[19] = 1.205           # r knee
    q[20] = 1.205           # l knee
    q[21] = 1.25            # r shoulder 
    q[22] = -1.25           # l shoulder
    q[23] = -0.71           # r ancle
    q[24] = -0.71           # l ancle
    q[25] = 0               # r elbow
    q[26] = 0               # l elbow
    q[27] = 0               # r ancle
    q[28] = 0               # l ancle
    q[29] = 0.78            # r elbow
    q[30] = -0.78           # l elbow
    q[31] = 1.571           # r wrist
    q[32] = 1.571           # l wrist
    q[33] = 0               # r wrist
    q[34] = 0               # l wrist
    q[35] = 0               # r wrist
    q[36] = 0               # l wrist

    qd = np.zeros(36)

    return np.hstack((q,qd))

def RPYValkyrieFixedPointTorque():
    """
    Return a set of torque commands that will keep the Valkyrie
    humanoid (approximately) fixed at the above initial state.
    
    Adopted from drake/examples/valkyrie/valkyrie_constants.cc.
    """
    tau = np.asarray([
      0, 54.07374714, -1.16973414,
      1.89429714, 
      
      3.778290679, -8.104844333, -1.370804286, 
      2.345797901, -0.3205054571, -0.2609708356, -0.1427544212, 
      
      3.778290679, 8.104844333, -1.370804286, 
      -2.345797901, -0.3205054571, 0.2609708356, 0.1427544212,
      
      0.0009084321844, 12.02429585, -10.18358769, -118.6322523, 52.87796422, 0.2418568986, 
      0.0009084320108, -11.43386868, -10.22606335, -116.9452938, 52.24348208, 0.2418569007])

    return tau

def ValkyrieFixedPointTorque():
    """
    Return torques that hold the robot (approximately) at the fixed position
    expressed by ValkyrieFixedPointState().
    """
    tau = np.zeros(30)
    tau[0] = 1.894    # spine
    tau[1] = 54.1    # spine
    tau[2] = -1.2    # spine
    tau[3] = 0    # l hip
    tau[4] = -11.4    # l hip
    tau[5] = 10.2    # l hip
    tau[6] = -117    # l knee
    tau[7] = 52.2    # l ancle
    tau[8] = 0.24    # l ancle
    tau[9] = 0    # r hip
    tau[10] = 12.02   # r hip 
    tau[11] = -10.2   # r hip
    tau[12] = -117   # r knee
    tau[13] = 52.9   # r ancle
    tau[14] = 0.24   # r ancle
    tau[15] = 0   # l shoulder
    tau[16] = 0   # l shoulder
    tau[17] = 0   # l shoulder
    tau[18] = 0   # l elbow
    tau[19] = 0   # l wrist
    tau[20] = 0   # l wrist
    tau[21] = 0   # l wrist
    tau[22] = 0   # r shoulder
    tau[23] = 0   # r shoulder
    tau[24] = 0   # r shoulder
    tau[25] = 0   # r elbow
    tau[26] = 0   # r wrist
    tau[27] = 0   # r wrist
    tau[28] = 0   # r wrist
    tau[29] = 0   # neck

    return tau
    

def RPYValkyrieFixedPointState():
    """
    Return a reasonable initial state for the Valkyrie humanoid. 
    Adopted from drake/examples/valkyrie/valkyrie_constants.cc.
    """
  
    # First six variables are spatial position of the floating base (x,y,z,r,p,y).
    # The next 30 variables are joint angles.
    q = np.asarray([
            0, 0, 1.025, 0, 0, 0,      \
            0, 0, 0, 0, 0.300196631343025, 1.25, 0, 0.785398163397448, \
            1.571, 0, 0, 0.300196631343025, -1.25, 0, -0.785398163397448, \
            1.571, 0, 0, 0, 0, -0.49, 1.205, -0.71, 0, 0, 0, -0.49, 1.205, -0.71, 0])
 
    # The first six variables are spatial velocities of the floating base.
    # The next 30 variables are joint velocities.
    qd = np.zeros(q.shape)

    return np.hstack((q,qd))

def RPYValkyrieFixedPointTorque():
    """
    Return a set of torque commands that will keep the Valkyrie
    humanoid (approximately) fixed at the above initial state.
    
    Adopted from drake/examples/valkyrie/valkyrie_constants.cc.
    """
    tau = np.asarray([
      0, 0, 0, 0, 0, 0, 0, 54.07374714, -1.16973414,                            \
      1.89429714, 3.778290679, -8.104844333, -1.370804286, 2.345797901,         \
      -0.3205054571, -0.2609708356, -0.1427544212, 3.778290679, 8.104844333,    \
      -1.370804286, -2.345797901, -0.3205054571, 0.2609708356, 0.1427544212,    \
      0.0009084321844, 12.02429585, -10.18358769, -118.6322523, 52.87796422,    \
      0.2418568986, 0.0009084320108, -11.43386868, -10.22606335, -116.9452938,  \
      52.24348208, 0.2418569007])

    return tau

def ComTrajectory(time):
    """
    Return a desired center of mass position and velocity for the given timestep.
    """

    # for now we'll consider this trajectory to be a straight line connecting 
    # two points

    t_switch = 1.0
    if time <= t_switch:
        x_com =  [0.0, (0.1/t_switch)*time, 1.0]
        xd_com = [0.0, 0.1/t_switch, 0.0]
    else:
        x_com = [0.0, 0.1, 1.0]
        xd_com = [0.0, 0.0, 0.0]

    return (x_com, xd_com)

def FootTrajectory(time):
    """
    Return a desired position and velocity for the right foot at the given
    timestep.
    """
    t_switch = 1.0
    if time <= t_switch:
        x_foot = [-0.075,-0.153,0.0827]
        xd_foot = [0.0, 0.0, 0.0]
    else:
        x_foot = [-0.075+0.5*(time-t_switch),-0.153,0.0827]
        xd_foot = [0.5, 0.0, 0.0]

    return (x_foot, xd_foot)

