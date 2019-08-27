##
#
# Helper functions for simulating a Valkyrie humanoid. 
#
##
import numpy as np
import time
from pydrake.all import *

def list_joints_and_actuators(robot):
    """
    Run through all the joints and actuators in the robot model 
    and print their names and indeces.
    """
    for i in range(robot.num_actuators()):
        joint_actuator = robot.get_joint_actuator(JointActuatorIndex(i))
        joint = joint_actuator.joint()

        print("Actuator [%s] acts on joint [%s]" % (joint_actuator.name(),joint.name()))

def AtlasFixedPointState():
    """
    Return a reasonable initial state for the Atlas humanoid, where
    the orientation is expressed in quaternions.
    """

    q = np.array([1,#9.28905836e-01,
                    0,#2.39236370e-03,
                    0,#-3.70963058e-01,
                    0,#4.98080637e-03,
                    0,#6.48422542e-02,
                    0,#1.95888551e-04,
                    0.75,#7.29813908e-01,
                    -2.27744518e-02,
                    -3.45599963e-02,
                    1.76481589e-02,
                    0,#+8.13731964e-01,
                    -2.83443570e-02,
                    9.34774007e-03,
                    -2.30525571e-02,
                    -0.65,#2.09538752e-01,
                    -0.65,#2.13166012e-01,
                    -3.33203785e-02,
                    -5.43372997e-01,
                    4.61125916e-02,
                    1.34741595e+00,
                    1.34665731e+00,
                    -4.19148641e-01,
                    4.47254631e-01,
                    -0.7,#-7.95624277e-01,
                    -0.7,#7.97017894e-01,
                    2.43621698e-01,
                    2.40396297e-01,
                    -4.01813646e-03,
                    4.65558086e-03,
                    4.66228967e-01,
                    -4.81489651e-01,
                    -3.00416436e-02,
                    -3.36320600e-02,
                    1.66319502e-02,
                    -1.16896875e-02,
                    -4.90277381e-03,
                    -4.59190158e-03])
    qd = np.zeros(36)

    return np.hstack((q,qd))

def MBP_RBT_joint_angle_map(multi_body_plant, rigid_body_tree):
    """
    Return a matrix X such that the joint angles (excluding floating base)
    of the rigid-body tree can be given as q_RBT = X*q_MBT, where q_MBT is
    the joint angles (excluding floating base) of the multi-body plant.
    """
    assert multi_body_plant.num_actuators() == rigid_body_tree.get_num_actuators()
    
    B_rbt = rigid_body_tree.B
    B_mbp = multi_body_plant.MakeActuationMatrix()

    X = np.dot(B_rbt[6:,:],B_mbp[6:,:].T)

    return(X)


def RPYAtlasFixedPointState():
    """
    Return a reasonable initial state for the Atlas humanoid. 
    """
    q = np.asarray([6.48422542e-02,  1.95888551e-04,  7.29813908e-01,  1.03295189e-03,
        -7.59904053e-01,  1.03114091e-02, -2.27744518e-02,  8.13731964e-01,
        -2.30525571e-02, -3.33203785e-02, -3.45599963e-02, -2.83443570e-02,
        2.09538752e-01,  1.34741595e+00, -7.95624277e-01, -4.01813646e-03,
        -4.19148641e-01,  2.43621698e-01,  4.66228967e-01, -3.00416436e-02,
        1.66319502e-02, -4.90277381e-03,  4.61125916e-02,  1.76481589e-02,
        9.34774007e-03,  2.13166012e-01,  1.34665731e+00, -7.97017894e-01,
        4.65558086e-03,  4.47254631e-01,  2.40396297e-01, -4.81489651e-01,
        -3.36320600e-02, -1.16896875e-02, -4.59190158e-03, -5.43372997e-01])
    qd = np.zeros(36)

    return np.hstack((q,qd))

