##
#
# Helper functions for simulating a Valkyrie humanoid. 
#
##
import numpy as np

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

