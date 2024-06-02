from scipy.sparse import spdiags
import random 
random.seed(10)

#import numpy as np

# def WR_LS_Denoise(y,lamda=0.01):
#     N = len(y)
#     lam = lamda
#     x = np.zeros((N-2,N), dtype=int)
#     for i in range(N-2):
#         x[i,i] = 1
#         x[i,i+1] = -2
#         x[i, i+2] = 1
#     z = np.eye(N,dtype = int) + lam*np.dot(np.transpose(x),x)
#     return np.dot(np.linalg.inv(z),y)


import cupy as cp

def WR_LS_Denoise(y,lamda=0.01):
    N = len(y)
    lam = lamda
    x = cp.zeros((N-2,N), dtype=int)
    for i in range(N-2):
        x[i,i] = 1
        x[i,i+1] = -2
        x[i, i+2] = 1
    z = cp.eye(N,dtype = int) + lam*cp.dot(cp.transpose(x),x)
    return cp.dot(cp.linalg.inv(z),y)