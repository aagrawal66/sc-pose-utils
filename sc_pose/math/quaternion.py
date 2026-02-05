""" Right-handed Scalar First (RSF) Quaternion Math utilities using python's numpy """
import math
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

def sscp_R3(v: NDArray) -> NDArray:
    """ Compute skew-symmetric cross product matrix for R^3 vectors """
    m_out   = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                        ])
    return m_out

def q_conj(q: NDArray) -> NDArray:
    """ Compute the conjugate of a RSF quaternion """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_norm(q: NDArray) -> NDArray:
    """ Normalize a RSF quaternion (4-vector form [w, x, y, z]) """
    n   = np.linalg.norm(q)
    if n < 1e-12:
        # avoid divide-by-zero; return identity
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def q2gibbs(q: NDArray) -> NDArray:
    """ Convert RSF quaternion to Gibbs vector """
    q   = q_norm(q)
    eps = 1e-12
    if abs(q[0]) < eps:
        raise ValueError("Gibbs vector undefined when scalar part ≈ 0 (θ ≈ π).") 
    return q[1:4] / q[0]

def gibbs2q(gibbs: NDArray) -> NDArray:
    """ Convert Gibbs vector to RSF quaternion """
    g_norm_sq  = np.dot(gibbs, gibbs)
    denom      = np.sqrt(1 + g_norm_sq)
    q_s        = 1.0 / denom
    q_v        = gibbs / denom
    q_out      = np.append([q_s], q_v)
    q_out      = q_norm(q_out)
    return q_out

def q_mult_shu(q2: NDArray, q1: NDArray) -> NDArray:
    """ 
    Compute the Shuster quaternion product of two RSF quaternions 
    q = q2 ⊗ q1
    """
    q1v         = q1[1:4]
    q1s         = q1[0]
    q2v         = q2[1:4]
    q2s         = q2[0]
    q2v_sscp    = sscp_R3(q2v)
    q_out       = np.append(
                                [q1s*q2s - np.dot(q2v, q1v)],
                                [q1s*q2v + q2s*q1v - q2v_sscp@q1v]
                            )
    q_out       = q_norm(q_out)
    # Note: q_mult_shu(q2, q1) == q_mult_ham(q1, q2)
    return q_out

def q_mult_ham(q2: NDArray, q1: NDArray) -> NDArray:
    """ 
    Compute the Hamilton quaternion product of two RSF quaternions 
    q = q2 ∘ q1
    """
    
    q1v         = q1[1:4]
    q1s         = q1[0]
    q2v         = q2[1:4]
    q2s         = q2[0]
    q2v_sscp    = sscp_R3(q2v)
    q_out       = np.append(
                                [q1s*q2s - np.dot(q2v, q1v)],
                                [q1s*q2v + q2s*q1v + q2v_sscp@q1v]
                            )
    q_out       = q_norm(q_out)
    # Note: q_mult_shu(q2,q1) == q_mult_ham(q1, q2)
    return q_out

def q_error_shu(q2: NDArray, q1: NDArray) -> NDArray:
    """ Compute the quaternion error between two RSF quaternions """
    q_err   = q_mult_shu(q2, q_conj(q1))
    q_err   = q_norm(q_err)
    return q_err

def q_error_ham(q2: NDArray, q1: NDArray) -> NDArray:
    """ Compute the quaternion error between two RSF quaternions """
    q_err   = q_mult_ham(q2, q_conj(q1))
    q_err   = q_norm(q_err)
    return q_err

def q2trfm(q: NDArray) -> NDArray:
    """ Convert an RSF quaternion to a transformation matrix (passive rotation), which is transpose of a rotation matrix """
    q       = q_norm(q)
    qv      = q[1:4]
    qs      = q[0]
    qv_sscp = sscp_R3(qv)
    Trfm    = (np.eye(3) + 2*qs*qv_sscp + 2*qv_sscp@qv_sscp).T
    return Trfm

def q2rotm(q: NDArray) -> NDArray:
    """ Converts an RSF quaternion to a rotation matrix (active rotation), not a transformation matrix """
    q       = q_norm(q)
    qv      = q[1:4]
    qs      = q[0]
    qv_sscp = sscp_R3(qv)
    Rm      = np.eye(3) + 2*qs*qv_sscp + 2*qv_sscp@qv_sscp
    return Rm

#TODO: implement rotm2q
def rotm2q(R: NDArray) -> NDArray:
    """ Convert an activate rotation matrix to an RSF quaternion with the the trace-based (Shoemake) method for numerical stability """
    R           = np.array(R, dtype = float)
    trace       = np.trace(R)
    if trace > 0:
        S   = np.sqrt(trace + 1.0) * 2  # S = 4*w
        w   = 0.25 * S
        x   = (R[2, 1] - R[1, 2]) / S
        y   = (R[0, 2] - R[2, 0]) / S
        z   = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S   = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4*x
        w   = (R[2, 1] - R[1, 2]) / S
        x   = 0.25 * S
        y   = (R[0, 1] + R[1, 0]) / S
        z   = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S   = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4*y
        w   = (R[0, 2] - R[2, 0]) / S
        x   = (R[0, 1] + R[1, 0]) / S
        y   = 0.25 * S
        z   = (R[1, 2] + R[2, 1]) / S
    else:
        S   = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4*z
        w   = (R[1, 0] - R[0, 1]) / S
        x   = (R[0, 2] + R[2, 0]) / S
        y   = (R[1, 2] + R[2, 1]) / S
        z   = 0.25 * S
    q   = q_norm( np.array([w, x, y, z]) )
    return q

def q2theta(q: NDArray) -> float:
    """ Compute the rotation angle (radians) from a RSF quaternion using euler axis and angle representation
    # q = [q_s, q_v] = [cos(theta/2), sin(theta/2) * axis]
    """
    return 2 * np.arccos(q_norm(q)[0])

def Rot_x(angle: float):
    """ Compute canonical rotation matrix for rotation about x-axis from an angle in radians, active rotation """
    c, s    = np.cos(angle), np.sin(angle)
    R_x       = np.array([
                            [1, 0, 0],
                            [0, c, -s], 
                            [0, s, c]
                        ])
    return R_x

def Rot_y(angle: float):
    """ Compute canonical rotation matrix for rotation about y-axis from an angle in radians, active rotation """
    c, s    = np.cos(angle), np.sin(angle)
    R_y       = np.array([
                            [c, 0, s], 
                            [0, 1, 0], 
                            [-s, 0, c]
                        ])
    return R_y

def Rot_z(angle: float):
    """ Compute canonical rotation matrix for rotation about z-axis from an angle in radians, active rotation """
    c, s    = np.cos(angle), np.sin(angle)
    R_z       = np.array([
                            [c, -s, 0], 
                            [s, c, 0], 
                            [0, 0, 1]
                        ])
    return R_z

def Rot_xyz(
                    roll : float,
                    pitch : float,
                    yaw : float
            ):
    """ Compute the active rotation matrix from the Euler angles in radians for a XYZ sequence where we rotate x first by rolling, then pitch by rotating about y, and then yawing about Z """
    R_x     = Rot_x(roll)
    R_y     = Rot_y(pitch)
    R_z     = Rot_z(yaw)
    R_xyz   = R_z @ R_y @ R_x
    return R_xyz

def q2rotv_logmap(q: NDArray, eps: float = 1e-12) -> NDArray:
    """
    Logarithmic map from unit quaternion to rotation vector (exponential-map parameters),
    with a branch choice that returns the *minimal-angle* rotation vector (theta ∈ (−π, π])
   
    - rotv magnitude = rotation angle in radians
    core equations (unit q):
        v       = [x, y, z]
        v_norm  = ||v||
        theta   = 2 * atan2(v_norm, w)
        rotv    = (theta / v_norm) * v  if v_norm > 0
        rotv    = 0 if v_norm = 0

    Minimal-angle branch:
    - q and -q represent the same rotation.
    - If w < 0, flip sign: q := -q  ⇒  theta becomes in [0, π] (shortest arc).
    - Additionally, wrap theta into (−π, π] just in case
    """
    # ensure unit quaternion
    q   = q_norm(q)

    # Choose shortest-arc representative: enforce w >= 0 (since q ≡ -q)
    if q[0] < 0.0:
        q   = -q

    w       = q[0]
    v       = q[1:]
    v_norm  = np.linalg.norm(v)

    if v_norm < eps:
        # small-angle stable branch:
        # For w ~ 1 and v small, rotv ≈ 2*v (since sin(theta/2) ~ theta/2)
        return 2.0 * v

    # angle (in [0, pi] after the w>=0 convention)
    theta       = 2.0 * np.arctan2(v_norm, w)

    # wrap to (-pi, pi] (defensive; with w>=0, theta should already be in [0, pi])
    if theta > np.pi:
        theta   -= 2.0 * np.pi

    rotv        = (theta / v_norm) * v
    return rotv


def rotv2q_expmap(rotv: NDArray, eps: float = 1e-12) -> NDArray:
    """ 
    Exponential map from rotation vector (R^3) to an RSF unit quaternion
    Let rotv ∈ R^3 be the rotation vector (axis * angle)
    - theta = ||rotv||
    The exponential map from so(3) to SO(3), expressed as a quaternion, is
    - q = exp( 0.5 * rotv )
    If theta > 0:
        w   = cos(theta / 2)
        xyz = (sin(theta / 2) / theta) * rotv
    If theta = 0:
        q = [1, 0, 0, 0]
    Small-angle approximation (theta → 0):
        sin(theta / 2) / theta ≈ 1/2 − theta^2 / 48 + O(theta^4)
        w   ≈ 1 − theta^2 / 8
        xyz ≈ 0.5 * rotv
    """
    theta  = np.linalg.norm(rotv)
    if theta < eps:
        # Small-angle stable branch (series expansion)
        # sin(theta/2)/theta ~ 1/2 - theta^2/48 + HOT
        # So vector part ~ 0.5*r, scalar part ~ 1 - theta^2/8
        half    = 0.5
        w       = 1.0 - (theta * theta) / 8.0
        xyz     = half * rotv
        q       = np.array([w, xyz[0], xyz[1], xyz[2]], dtype = float)
        q       = q_norm(q)
        return q