from __future__ import division

import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, \
 RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop, GaussianBlur
from torch.utils.data import Dataset, DataLoader


import os
import copy

import data_transform.transforms as extended_transforms
import data_transform.modified_randaugment as rand_augment

import torch
import torch.nn as nn
import torch.nn.functional as F 

import math

'''
NPR functions based on sinkhorn-knopp, 
https://github.com/ChengHan111/DNC
'''
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# K * #Guass as input
@torch.no_grad()
def AGD_torch_no_grad_gpu(M, maxIter=20, eps=0.05):
    M = M.t() # [#Guass, K]
    p = M.shape[0] # #Guass
    n = M.shape[1] # K 
    
    X = torch.zeros((p,n), dtype=torch.float64).to(M.device)

    r = torch.ones((p,), dtype=torch.float64).to(M.device) / p # .to(L.device) / K
    c = torch.ones((n,), dtype=torch.float64).to(M.device) / n # .to(L.device) / B 先不要 等会加上

    max_el = torch.max(abs(M)) #np.linalg.norm(M, ord=np.inf)
    gamma = eps/(3*math.log(n)) 

    A = torch.zeros((maxIter, 1), dtype=torch.float64).to(M.device) #init array of A_k
    L = torch.zeros((maxIter, 1), dtype=torch.float64).to(M.device) #init array of L_k

    # set initial values for APDAGD
    L[0,0] = 1; #set L_0

    #set starting point for APDAGD
    y = torch.zeros((n+p, maxIter), dtype=torch.float64).to(M.device) #init array of points y_k for which usually the convergence rate is proved (eta)
    z = torch.zeros((n+p, maxIter), dtype=torch.float64).to(M.device) #init array of points z_k. this is the Mirror Descent sequence. (zeta)    
    j = 0
    # main cycle of APDAGD
    for k in range(0,(maxIter-1)):
                         
        L_t = (2**(j-1))*L[k,0] #current trial for L            
        a_t = (1  + torch.sqrt(1 + 4*L_t*A[k,0]))/(2*L_t) #trial for calculate a_k as solution of quadratic equation explicitly
        A_t = A[k,0] + a_t; #trial of A_k
        tau = a_t / A_t; #trial of \tau_{k}     
        x_t = tau*z[:,k] + (1 - tau)*y[:,k]; #trial for x_k
        
        lamb = x_t[:n,]
        mu = x_t[n:n+p,]    
        
        # 1) [K,1] * [1, #Gauss] --> [K, #Gauss].T -->[#Gauss, K]; 2) [K, 1] * [#Guass, 1].T --> [K, #Guass]--.T--> [#Guass, K]
        M_new = -M - torch.matmul(lamb.reshape(-1,1).to(M.device), \
                                  torch.ones((1,p), dtype=torch.float64).to(M.device)).T \
        - torch.matmul(torch.ones((n,1), dtype=torch.float64).to(M.device),\
                       mu.reshape(-1,1).T.to(M.device)).T

        X_lamb = torch.exp(M_new/gamma)
        sum_X = torch.sum(X_lamb)
        X_lamb = X_lamb/sum_X
        grad_psi_x_t = torch.zeros((n+p,), dtype=torch.float64).to(M.device)
        grad_psi_x_t[:p,] = r - torch.sum(X_lamb, axis=1)
        grad_psi_x_t[p:p+n,] = c - torch.sum(X_lamb, axis=0).T

        #update model trial
        z_t = z[:,k] - a_t*grad_psi_x_t #trial of z_k 
        y_t = tau*z_t + (1 - tau)*y[:,k] #trial of y_k

        #calculate function \psi(\lambda,\mu) value and gradient at the trial point of y_{k}
        lamb = y_t[:n,]
        mu = y_t[n:n+p,]           
        M_new = -M - torch.matmul(lamb.reshape(-1,1).to(M.device), \
                                  torch.ones((1,p), dtype=torch.float64).to(M.device)).T \
        - torch.matmul(torch.ones((n,1), dtype=torch.float64).to(M.device),\
                       mu.reshape(-1,1).T.to(M.device)).T
        Z = torch.exp(M_new/gamma)
        sum_Z = torch.sum(Z)

        X = tau*X_lamb + (1-tau)*X #set primal variable 
            # break
             
        L[k+1,0] = L_t
        j += 1
    
    X = X.t()

    indexs = torch.argmax(X, dim=1)
    G = F.gumbel_softmax(X, tau=0.5, hard=True)

    return G.to(torch.float32), indexs # change into G as well 

# first initialize prototypes

class npr(nn.Module):
    def __init__(self, cluster=4, classnum=8, embed=1280, trainable=False):
        super(npr,self).__init__()
        self.prototypes = nn.Parameter(torch.zeros(classnum, cluster, \
            embed), requires_grad=trainable)
        self.classnum = classnum
        self.cluster = cluster
        self.embed = embed
        self.trainable = trainable
        trunc_normal_(self.prototypes, std=0.02)
        
    def compute_and_update_proto(self):
        
        pass
                                            
    def update_proto(self, new_proto):
        # normalize new_proto
        new_proto = F.normalize(new_proto, p=2, dim=-1) # clasnum, cluster, embed
        # update
        self.prototypes = nn.Parameter(new_proto, requires_grad=self.trainable)
        
    def forward(self,x):
        
        return y,y2             

