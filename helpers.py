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

    q = np.array([ 9.97302180e-01,  5.55185393e-04, -7.34314691e-02,  5.95320986e-04,
       -9.30385970e-03, -4.09673631e-03,  8.09777460e-01, -2.36118176e-02,
       -3.17723010e-02,  1.40016791e-02,  1.66060065e-01, -3.10627942e-02,
        1.71959561e-02, -2.34602348e-02, -4.49754539e-01, -4.50475437e-01,
       -3.07507399e-02, -5.42326307e-01,  4.28314425e-02,  1.23945441e+00,
        1.24018020e+00, -3.91502463e-01,  4.17143616e-01, -6.42859361e-01,
       -6.42831351e-01,  2.42515748e-01,  2.39086139e-01,  2.50276879e-02,
       -1.61994843e-02,  4.74058368e-01, -4.90065384e-01, -3.00399838e-02,
       -3.36413202e-02,  1.72228821e-02, -1.23430846e-02, -4.90180126e-03,
       -4.59238965e-03])
    
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

