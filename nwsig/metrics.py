
#We define different metrics on the space of paths, which are used to define the Nadaraya-Watson estmator.
import numpy as np
import iisignature as ii
import torch
import sigkernel


def supremum_metric(y_true, y_pred):
    """
    Compute the supremum metric (L-infinity norm) between true and predicted values.
    """
    return np.max(np.abs(y_true - y_pred))

def euclidean_metric(y_true, y_pred):
    """
    This can be used as truncated signature metric, when the input is the signature already
    """
    return np.linalg.norm(y_true - y_pred)
def trunc_sig_metric(y_true, y_pred, truncation_level=3):

    return np.linalg.norm(ii.sig(y_true, truncation_level) - ii.sig(y_pred, truncation_level))

def full_sig_metric(y_true, y_pred,static_kernel_spec='linear',sigma=1,scaling=1,subsampling_dual = 1,L =1,N1=1):
    """
    Using signature-kernel PDE
    """
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)
    y_true_torch = torch.from_numpy(y_true.reshape(1,y_true.shape))
    y_pred_torch = torch.from_numpy(y_pred.reshape(1,y_pred.shape))
    dist = signature_kernel.compute_kernel(y_true_torch, y_true_torch).numpy()-2*signature_kernel.compute_kernel(y_true_torch, y_pred_torch).numpy()+signature_kernel.compute_kernel(y_pred_torch, y_pred_torch).numpy()
    
    return np.sqrt(dist)
def MSE(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)
#Todo: p-variation metric/alpha-Hölder metrc

