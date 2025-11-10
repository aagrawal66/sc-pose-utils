""" Right-handed Scalar First (RSF) Quaternion Math utilities using python's numpy """

import numpy as np

def sscp_R3(v):
    """ Compute skew-symmetric cross product matrix for R^3 vectors """
    m_out   = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                        ])
    return m_out

def q_conj(q):
    """ Compute the conjugate of a RSF quaternion """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_norm(q):
    """ Normalize a RSF quaternion (4-vector form [w, x, y, z]) """
    n   = np.linalg.norm(q)
    if n < 1e-12:
        # avoid divide-by-zero; return identity
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def q2gibbs_vec(q):
    """ Convert RSF quaternion to Gibbs vector """
    return q[1:4] / q[0]

def q_mult_shu(q2, q1):
    """ Compute the Shuster quaternion product of two RSF quaternions """
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
    return q_out
    
def q_mult_ham(q2, q1):
    """ Compute the Hamilton quaternion product of two RSF quaternions """
    
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

    return q_out

def q_error_shu(q2, q1):
    """ Compute the quaternion error between two RSF quaternions """
    q_err   = q_mult_shu(q2, q_conj(q1))
    q_err   = q_norm(q_err)
    return q_err

def q_error_ham(q2, q1):
    """ Compute the quaternion error between two RSF quaternions """
    q_err   = q_mult_ham(q2, q_conj(q1))
    q_err   = q_norm(q_err)
    return q_err

def q2trfm(q): 
    """ Convert an RSF quaternion to a transformation matrix (passive rotation), which is transpose of a rotation matrix """ 
    q       = q_norm(q)
    qv      = q[1:4]
    qs      = q[0]
    qv_sscp = sscp_R3(qv)
    Trfm    = (np.eye(3) + 2*qs*qv_sscp + 2*qv_sscp@qv_sscp).T
    return Trfm

def q2rotm(q):
    """ Converts an RSF quaternion to a rotation matrix (active rotation), not a transformation matrix """
    q       = q_norm(q)
    qv      = q[1:4]
    qs      = q[0]
    qv_sscp = sscp_R3(qv)
    Rm      = np.eye(3) + 2*qs*qv_sscp + 2*qv_sscp@qv_sscp
    return Rm