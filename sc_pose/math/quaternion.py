""" Right-handed Scalar First (RSF) Quaternion Math utilities using python's numpy """

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