""" Pytorch Pose Error Metrics, based on SPEED+ Error Metrics """
import torch
# torch batch versions
def torch_batch_E_T(tr, tr_hat):
        """ This function computes the 2norm of difference between two batches of torch translation vectors """
        err_t   = torch.norm(tr - tr_hat, dim = 1) 
        # shape of err_t is (N,)
        return err_t
    
def torch_batch_E_TN(tr, tr_hat):
    """ This function computes the 2norm of difference between two batches of torch translation vectors """
    err_tn  = torch.norm(tr - tr_hat, dim = 1) / torch.norm(tr, dim = 1)
    # shape of err_tn is (N,)
    return err_tn

def torch_batch_E_R(quat, quathat):
    """ This function computes the rotational error between two batches of torch quaternions (non-robust) """
    quat        = quat / torch.norm(quat, dim = 1, keepdim = True)
    quathat     = quathat / torch.norm(quathat, dim = 1, keepdim = True)
    abs_dotprod = torch.abs( torch.sum(quat * quathat, dim = 1) )
    err_r       = 2 * torch.acos( torch.clamp(abs_dotprod, -1, 1) )
    # shape of err_r is (N,)
    return err_r