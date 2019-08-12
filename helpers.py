##
#
# Helper functions for simulating a Valkyrie humanoid. 
#
##
import numpy as np
import time
from pydrake.all import *

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

