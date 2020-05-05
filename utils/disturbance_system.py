# Defines a class that can be used to generate external wrench disturbances on 
# a plant. 

import numpy as np
from pydrake.all import *

class DisturbanceSystem(LeafSystem):
    def __init__(self, plant, body_name, spatial_force, time, duration):
        """
        This system generates a spatial force (spatial_force) of the given value, and
        sets up an output that will apply this force to the given body (body_name)
        of a plant (plant) for a given duration (duration) at the given time (time). 
        """
	LeafSystem.__init__(self)
	self.nv = plant.num_velocities()

        # Specify which body the force will be applied to
	self.target_body_index = plant.GetBodyByName(
	    body_name).index()

        # Record force, duration, and time of disturbance
        self.t = time
        self.wrench = spatial_force.reshape(6)   # spatial_force/wrench = [torque; force]
        self.dur = duration

        # Set up output port
        self.DeclareAbstractOutputPort(
	    "spatial_forces_vector",
	    lambda: AbstractValue.Make(
		VectorExternallyAppliedSpatialForced()),
	    self.DoCalcAbstractOutput)

    def DoCalcAbstractOutput(self, context, y_data):
	test_force = ExternallyAppliedSpatialForce()
	test_force.body_index = self.target_body_index
	test_force.p_BoBq_B = np.zeros(3)

        if self.t <= context.get_time() and context.get_time() <= self.t+self.dur:
            # apply the external force at the designated times
            test_force.F_Bq_W = SpatialForce(tau=self.wrench[0:3],
                    f=self.wrench[3:6])
        else:
            # don't apply any force otherwise
            test_force.F_Bq_W = SpatialForce(tau=np.array([0.,0.,0.]),
                    f=np.array([0.,0.,0.]))

	y_data.set_value(VectorExternallyAppliedSpatialForced([
	    test_force]))
