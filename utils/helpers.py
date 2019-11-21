##
#
# Helper functions related to simulating a Valkyrie Humanoid
#
##

import time
import numpy as np
import cdd

def AddQuadraticConstraint(mp, Q, b, c, x):
    """
    Add a quadratic constraint to the given Drake mathematical program mp,

        x'*Q*x + b'*x + c <= 0.

    Works by reformulating the constraint as a Lorenz cone constraint,
    which Drake's MathematicalProgram interface can handle. 
    """

    # Note and check relevant dimensions
    n = x.shape[0]
    assert x.shape == (n,1) , "x must by nx1 numpy array"
    assert Q.shape == (n,n) , "Q must be nxn numpy array"
    assert b.shape == (n,1) , "b must be nx1 numpy array"
    assert isinstance(c, (int, long, float)) or c.size == 1 , "c must be a scalar"

    # Take cholesky decomposition Q = A'*A
    A = np.linalg.cholesky(Q).T

    # Compute relevant inverses
    Ainv = np.linalg.inv(A)
    Qinv = np.dot(Ainv,Ainv.T)

    # Quick feasibility check
    bQinvb = np.dot(np.dot(b.T,Qinv),b)
    assert 0.5*bQinvb >= c , "Quadratic constraint probably infeasible, 1/2*b'*Q^{-1}*b - c < 0"

    # Now we'll define a linear transformation from x to z, 
    #    z = Fx + g 
    # such that we have the equivalent constraint
    #
    #    z_0 >= \sqrt{ z_1^2 + z_2^2 + ... }.

    F = np.vstack([ np.zeros((1,n)),  # [ 0 ]
                    A             ])  # [ A ]

    g = np.vstack([ np.sqrt( 0.5*bQinvb - c),  # [ \sqrt{ 1/2*b'*Q^{-1}*b -c } ]
                    0.5*np.dot(Ainv.T,b)])     # [        1/2*(A')^{-1}*b      ]

    z = np.dot(F,x) + g

    # Add the constraint to the MathematicalProgram
    return mp.AddLorentzConeConstraint(z)

def AddForwardEulerDynamicsConstraint(mp, A, B, x, u, xnext, dt):
    """
    Add a dynamics constraint to the given Drake mathematical program mp, represinting
    the euler dynamics:

        xnext = x + (A*x + B*u)*dt,

    where x, u, and xnext are symbolic variables.
    """
    n = A.shape[0]
    Aeq = np.hstack([ (np.eye(n)+A*dt), B*dt, -np.eye(n) ])
    beq = np.zeros((n,1))
    xeq = np.hstack([ x, u, xnext])[np.newaxis].T

    return mp.AddLinearEqualityConstraint(Aeq,beq,xeq)

def S(a):
    """
    Return the 3x3 cross product matrix 
    such that S(a)*b = a x b.
    """
    assert a.shape == (3,) , "Input vector is not a numpy array of size (3,)"
    S = np.asarray([[ 0.0 ,-a[2], a[1] ],
                    [ a[2], 0.0 ,-a[0] ],
                    [-a[1], a[0], 0.0  ]])

    return S

def regularize_linear_constraint(A,b):
    """
    Given a linear constraint of the form Ax <= b, return a new linear 
    constraint A'x <= b' which removes all of the redundant constraints.
    """

    assert type(A) == np.ndarray, "A must be a numpy array"
    assert type(b) == np.ndarray, "b must be a numpy array"
    assert A.shape[0] == b.shape[0], "Invalid constraint dimensions %s and %s" % (A.shape, b.shape)

    n = A.shape[1]   # size of x
    
    # Convert to cdd H-representation
    H_np = np.hstack([A,b])
    H = cdd.Matrix(H_np, number_type='float')
    H.rep_type = cdd.RepType.INEQUALITY

    # Remove redundancies
    H.canonicalize()

    # Convert back to original numpy 
    H_np = np.asarray(H)
    A_new = H_np[:,0:n]
    b_new = H_np[:,n][np.newaxis].T

    return A_new, b_new

def face_to_span(A):
    """
    Convert a polyhedral cone from face form

        C = {x | Ax <= 0}

    to span form

        C = {Vz | z >= 0}.
    """
    # H-representation [b -A], where Ax <= b
    H = np.hstack([np.zeros((A.shape[0],1)),-A])

    mat = cdd.Matrix(H, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    P = cdd.Polyhedron(mat)

    # V-representation [t V], where t=0 for rays
    g = P.get_generators()
    tV = np.array(g)

    assert np.all(tV[:,0] == 0), "Not a cone!"

    rays = []
    for i in range(tV.shape[0]):
        if i not in g.lin_set:
            rays.append(tV[i,1:])
    V = np.asarray(rays).T

    return V

def span_to_face(V):
    """
    Convert a polyhedral cone from span form

        C = {Vz | z >= 0}.

    to face form
        
        C = {x | Ax <= 0}
    """
    # V-representation [t V], where t=0 for rays
    tV = np.hstack([np.zeros((V.shape[1],1)),V.T])
    
    mat = cdd.Matrix(tV, number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(mat)

    # H-representation [b -A], where Ax <= b
    ineq = P.get_inequalities()
    H = np.array(ineq)

    #assert np.all(H[:,0] == 0), "Ax <= b, but b is nonzero!"

    A = []
    for i in xrange(H.shape[0]):
        if i not in ineq.lin_set:
            A.append(-H[i,1:])

    return np.asarray(A)

def get_total_mass(tree):
    """
    Return the total mass for a given RigidBodyTree
    """
    m = 0
    for body in tree.get_bodies():
        spatial_inertia = body.get_spatial_inertia()
        body_mass = spatial_inertia[-1,-1]
        m += body_mass
    return m

def is_pos_def(matrix):
    """
    Return true if the given matrix (2d numpy array) is positive
    definite.
    """
    return np.all(np.linalg.eigvals(matrix) > 0)

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
      0, 0, 0, 0, 0, 0,
      0, 54.07374714, -1.16973414,
      1.89429714, 
      
      3.778290679, -8.104844333, -1.370804286, 
      2.345797901, -0.3205054571, -0.2609708356, -0.1427544212, 
      
      3.778290679, 8.104844333, -1.370804286, 
      -2.345797901, -0.3205054571, 0.2609708356, 0.1427544212,
      
      0.0009084321844, 12.02429585, -10.18358769, -118.6322523, 52.87796422, 0.2418568986, 
      0.0009084320108, -11.43386868, -10.22606335, -116.9452938, 52.24348208, 0.2418569007])

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
    qd = np.zeros(len(q))

    return np.hstack((q,qd))

