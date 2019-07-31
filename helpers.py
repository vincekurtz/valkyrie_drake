##
#
# Helper functions for simulating a Valkyrie humanoid. 
#
##
import numpy as np
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

