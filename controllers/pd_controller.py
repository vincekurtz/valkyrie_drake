##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import time
import numpy as np
from pydrake.all import *
from utils.helpers import *
from utils.walking_pattern_generator import *

class ValkyriePDController(VectorSystem):
    """
    A simple PD controller that regulates the robot to a nominal (standing) position.
    """
    def __init__(self, tree, plant, Kp=500.00, Kd=2.0):
        
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

        self.nominal_state = RPYValkyrieFixedPointState()  # joint angles and torques for nominal position
        self.tau_ff = RPYValkyrieFixedPointTorque()

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

    def ComputePDControl(self, q, qd, feedforward=True):
        """
        Map state [q,qd] to control inputs [u].
        """
        q_nom = self.nominal_state[0:self.np]
        qd_nom = self.nominal_state[self.np:]

        # compute torques to be applied in joint space
        if feedforward:
            tau = self.tau_ff + self.Kp*(q_nom-q) + self.Kd*(qd_nom - qd)
        else:
            tau = self.Kp*(q_nom-q) + self.Kd*(qd_nom - qd)

        # Convert torques to actuator space
        B = self.tree.B
        u = np.dot(B.T,tau)

        return u

    def DoCalcVectorOutput(self, context, state, unused, output):
        q,qd = self.StateToQQDot(state)

        u = self.ComputePDControl(q,qd,feedforward=True)
        output[:] = u
