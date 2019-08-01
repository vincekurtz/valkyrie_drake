##
#
# Example whole-body controllers for the Valkyrie robot.
#
##

import numpy as np
import time
from pydrake.all import *
from helpers import *

class ValkyrieController(VectorSystem):
    """
    PD Controller that attempts to regulate the robot to a fixed initial position
    """
    def __init__(self, robot):


        VectorSystem.__init__(self, 
                              robot.num_positions()+robot.num_velocities(),   # input is [q,v]
                              robot.num_actuators())                          # output is [tau]
        
        self.robot = robot

        # Nominal state
        self.nominal_state = ValkyrieFixedPointState()

    def DoCalcVectorOutput(self, leaf_context, state, unused, output):
        """
        Map from the state (q,qd) to output torques.
        """

        q = state[0:self.robot.num_positions()]
        v = state[self.robot.num_positions():]

        q_des = self.nominal_state[0:self.robot.num_positions()]
        v_des = self.nominal_state[self.robot.num_positions()]

        M, Cv, tauG, B, tauExt = ManipulatorDynamics(self.robot, q, v)

        context = self.robot.CreateDefaultContext()
        q_err = self.robot.MapQDotToVelocity(context, q_des-q)   # need to use this mapping since positions
                                                                 # use quaternions

        v_err = v_des - v

        # simple PD scheme to determine vd_Des
        vd_des = 100*v_err + 10000*q_err

        u = solve_joint_accel_QP(M,Cv,tauG,B,tauExt,vd_des)

        output[:] = u
