""" Pose Error Metrics, based on SPEED+ Error Metrics """
import numpy as np
from numpy.typing import NDArray

# local imports
from sc_pose.math.quaternion import q_error_shu

def rmse_calc(gt: NDArray, pred: NDArray):
    """ Calculate the Root Mean Squared Error (RMSE) between the ground truth and predicted values """
    rmse   = np.sqrt( np.mean( (gt - pred) **2) )
    return rmse

def norm_err_calc(gt: NDArray, pred: NDArray, norm_type: int = 2):
    """ Calculate the Lp norm between the ground truth and predicted values """
    norm_calc   = np.linalg.norm(gt - pred, ord = norm_type)
    return norm_calc

def E_T(tr: NDArray, tr_hat: NDArray):
    """ This function computes 2norm of difference between two translation vectors """
    e_tr    = np.linalg.norm(tr-tr_hat, ord = 2) 
    return e_tr

def E_TN(tr: NDArray, tr_hat: NDArray):
    """ This function computes normalized 2norm of difference between two translation vectors, the first vector is often the truth translation """
    err_tn  = E_T(tr, tr_hat) / np.linalg.norm(tr, ord = 2)
    return err_tn

def E_R(quat: NDArray, quathat: NDArray):
    """ This function computes rotational error by finding the angle of smallest rotation between the truth and estimated attitudes, robust form """
    quat        = quat / np.linalg.norm(quat)
    quathat     = quathat / np.linalg.norm(quathat)
    abs_dotprod = np.abs( np.dot(quat, quathat) )
    abs_dotprod = np.where(abs_dotprod > 1, np.abs(q_error_shu(quat, quathat)[0]), abs_dotprod)
    err_r       = 2 * np.arccos( abs_dotprod )
    return err_r

# batch versions
def batch_E_T(tr, tr_hat):
    """ This function computes the 2norm of difference between two batches of translation vectors"""
    err_t   = np.linalg.norm(tr - tr_hat, ord = 2, axis = 1)
    # shape of err_t is (N,)
    return err_t

def batch_E_TN(tr, tr_hat):
    """ This function computes the 2norm of difference between two batches of translation vectors """
    err_tn  = np.linalg.norm(tr - tr_hat, ord = 2, axis = 1) / np.linalg.norm(tr, ord = 2, axis = 1)
    # shape of err_tn is (N,)
    return err_tn

def batch_E_R(quat, quathat):
    """ This function computes the rotational error between two batches of quaternions (non-robust) """
    quat        = quat / np.linalg.norm(quat, axis = 1, keepdims = True)
    quathat     = quathat / np.linalg.norm(quathat, axis = 1, keepdims = True)
    abs_dotprod = np.abs( np.sum(quat * quathat, axis = 1) )
    err_r       = 2 * np.arccos( abs_dotprod )
    # shape of err_r is (N,)
    return err_r