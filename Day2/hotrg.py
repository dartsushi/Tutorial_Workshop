import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
from Initialization import Ising_Tc, initialize_Ising, impurity_Ising, initialize_Ising_mag
from loop_opt import normalize_tensor
from cft import cal_sdimension

def trace_tensor(A):
    return np.einsum("ijij",A)

def find_U(A,χ,cutoff=1e-14):
    four_tensor = ncon([A,A,np.conj(A),np.conj(A)],([-1,2,1,5],[-2,5,3,4],[-3,2,1,6],[-4,6,3,4]))
    shape = four_tensor.shape
    tmp = four_tensor.reshape(shape[0]*shape[1],-1)
    u,s,vt = np.linalg.svd(tmp,hermitian=False)
    s = s[s>cutoff]
    χ_cut = min(len(s),χ)
    u = u[:,:χ_cut]
    return u.reshape(shape[0],shape[1],-1)

def contract_vertical(A,χ,cutoff=1e-14):
    u = find_U(A,χ,cutoff=cutoff)
    return ncon([A,A,u,np.conj(u)],([2,-2,4,1],[3,1,5,-4],[4,5,-3],[2,3,-1]))

def contract_horizontal(A,χ,cutoff=1e-14):
    return contract_vertical(A.transpose(1,2,3,0),χ,cutoff=cutoff).transpose(3,0,1,2)

def HOTRG_step(A,χ,cutoff=1e-14):
    A_new = contract_vertical(A,χ,cutoff=cutoff)
    A_new = contract_horizontal(A_new,χ,cutoff=cutoff)
    A_new,_,norm = normalize_tensor(A_new,A_new)
    return A_new, norm

def cal_lnz(T,h,χ,RG_step=20):
    lnz = 0
    area = 4
    A = initialize_Ising_mag(T,h)
    A_new,_,norm = normalize_tensor(A,A)
    lnz += np.log(norm)/area
    for i in range(RG_step):
        A_new, norm = HOTRG_step(A_new,χ,cutoff=1e-12)
        area *= 4
        lnz += np.log(norm)/area
    return lnz


def contract_vertical_S(A,S,χ,cutoff=1e-14):
    u = find_U(A,χ,cutoff=cutoff)
    A_new = ncon([A,A,u,np.conj(u)],([2,-2,4,1],[3,1,5,-4],[4,5,-3],[2,3,-1]))
    S_up = ncon([A,S,u,np.conj(u)],([2,-2,4,1],[3,1,5,-4],[4,5,-3],[2,3,-1]))
    S_down = ncon([S,A,u,np.conj(u)],([2,-2,4,1],[3,1,5,-4],[4,5,-3],[2,3,-1]))
    return A_new, 0.5*(S_up+S_down)

def contract_horizontal_S(A,S,χ,cutoff=1e-14):
    A_new, S_new = contract_vertical_S(A.transpose(1,2,3,0),S.transpose(1,2,3,0),χ,cutoff=cutoff)
    return A_new.transpose(3,0,1,2), S_new.transpose(3,0,1,2)

def HOTRG_step_impurity(A,S,χ,cutoff=1e-14):
    A_new,S_new = contract_vertical_S(A,S,χ,cutoff=cutoff)
    A_new,S_new = contract_horizontal_S(A_new,S_new,χ,cutoff=cutoff)
    norm = trace_tensor(A_new)
    A_new /= norm
    S_new /= norm
    return A_new, S_new, norm

def magnetization(T,χ,h,RG_step=20):
    lnz = 0
    area = 1
    A_new = initialize_Ising_mag(T,h)
    S_new = impurity_Ising(T)
    for i in range(RG_step):
        A_new,S_new,norm = HOTRG_step_impurity(A_new,S_new,χ,cutoff=1e-14)
        area *= 4
        lnz += np.log(norm)/area
    return trace_tensor(S_new)

def impurity_expectation(A,S,χ,RG_step=20):
    lnz = 0
    area = 1
    A_new = A
    S_new = S
    for i in range(RG_step):
        A_new,S_new,norm = HOTRG_step_impurity(A_new,S_new,χ,cutoff=1e-14)
        area *= 4
        lnz += np.log(norm)/area
    return trace_tensor(S_new)
    
    
    
"""
  3
  |
1- -4
 [W]
2- -5
  |
  6
  
  
  2   4
  |   |
  [ h ]
  |   |
  1   3
"""

# renormalize the tensor horizontally and vertically one time each so that the new tensor become one-site impurity tensor.
def imputiry_two(A,W,χ,cutoff=1e-14):
    u = find_U(A,χ,cutoff=cutoff)
    A_new = ncon([A,A,u,np.conj(u)],([2,-2,4,1],[3,1,5,-4],[4,5,-3],[2,3,-1]))
    S_new = ncon([W,u,np.conj(u)],([1,2,-2,3,4,-4],[3,4,-3],[1,2,-1]))
    A_new,S_new = contract_horizontal_S(A_new,S_new,χ,cutoff=cutoff)
    return A_new, S_new


def twosite_impurity_expectation(A,W,χ,RG_step=20):
    lnz = 0
    area = 1
    A_new,S_new = imputiry_two(A,W,χ,cutoff=1e-14)
    for i in range(RG_step):
        A_new,S_new,norm = HOTRG_step_impurity(A_new,S_new,χ,cutoff=1e-12)
        area *= 4
        lnz += np.log(norm)/area
    return trace_tensor(S_new)


# O: on-site operator  H: two-site operator
def make_impurity(λ,χ,O,H):
    A = load_tensor(λ,χ)
    shape = A.shape
    I = np.identity(shape[-1])
    OI = 1/2*(np.tensordot(O,I,axes=0)+np.tensordot(I,O,axes=0))
    AA = ncon([A,np.conj(A)],([-1,-3,-5,-7,1],[-2,-4,-6,-8,1]))
    AA = AA.reshape(shape[0]**2,shape[1]**2,shape[2]**2,shape[3]**2)
    # energy terms as impurity tensors
    W_onsite = ncon([A,A,np.conj(A),np.conj(A),OI],([-1,-5,-7,1,2],[-3,1,-9,-11,3],[-2,-6,-8,4,5],[-4,4,-10,-12,6],[2,5,3,6]))
    W_onsite = W_onsite.reshape(shape[0]**2,shape[0]**2,shape[1]**2,shape[2]**2,shape[2]**2,shape[3]**2)
    W_coupling = ncon([A,A,np.conj(A),np.conj(A),H],([-1,-5,-7,1,2],[-3,1,-9,-11,3],[-2,-6,-8,4,5],[-4,4,-10,-12,6],[2,5,3,6]))
    W_coupling = W_coupling.reshape(shape[0]**2,shape[0]**2,shape[1]**2,shape[2]**2,shape[2]**2,shape[3]**2)
    #A: (left, up, right, down)
    return AA, W_onsite+W_coupling