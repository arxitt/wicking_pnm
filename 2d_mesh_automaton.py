# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:30:10 2021

@author: firo
"""

import networkx as nx
import scipy as sp
import scipy.sparse
import scipy.linalg
import scipy.ndimage
import numpy as np
from skimage.morphology import square

patm = 101.325E3 #Pa
g = 9.81 #m/s2
rho = 1000 #kg/m3
grid_size = 1E-3 #m
eta = 1E-5 #Pas
gamma = 72E-3 #N/m


# experimental input
h0 = 25E-3 #m, maximum height
C = 1.1E-3 #m/sqrt(s), wicking constant

r = 1E-4 #m
K0 = np.pi*r**4/8/eta/grid_size
pc0 = 2*gamma/r
# pc0 = h0*rho*g

domain_size = (70, 40)


def solve_pressure_field(p, mask, acts, inlets, K_mat, pg, pc):
    # TODO: solver is not the problem, get rid of copy by calculculating fluxes outside
    # consider dropping masking and get rid of the loop, only update parts of K_mat and A by keeping them alive outside
    # define RHS (boundary conditions)
    # TODO: always check signs of pressures
    p[:]= 0  
    # p[mask] = pg[mask]
    p[acts] = p[acts] - pc[acts] + patm  #TODO: add perturbation 
    p[inlets] = p[inlets] + patm 
    
    # define conductance (K_mat) and LHS (A) matrix

    A = - K_mat.copy()
    A[~mask, :] = 0
    A[:, ~mask] = 0
    np.fill_diagonal(A, -A.sum(axis=0))
     
    A[inlets,:] = 0
    A[inlets,inlets] = 1
     
    for i in acts:
         A[i,:] = 0
         A[i,i] = 1  
     
    A = A[mask,:]
    A = A[:,mask] 
      
    # solve equation system for pressure field
    p[mask] = np.linalg.solve(A, p[mask])- pg[mask]         
    p_mat = p-p[:,None]
       
    # get pore fluxes
    q_ij = -K_mat*p_mat
    q_i = q_ij.sum(axis=0)
    
    return q_i, p

# store filling state in numpy array v>=1:filled, 0:empty, 0<v<1 active
# dump this state to 3D array every once in a while -> result
# DONE: exctract actives and filled from this array and apply Laplace filter on it to get wetting force multiplicator at finger tips
# DONE: find good look up method for pore element coordinates from node indices (needed for gravity and update of filling array)
# include fiber orientation as non-isotropic weighting factor for wetting force
#  consider adding diagonal conductivity
#  estimate capillary pressure pc0 and conductivity from experiment
#  DONE: consider calculating conductivityx using filling state to better reflect disztance effect of dissipation
# include local drainage


def front_extraction(fill_mat):
    dilated = sp.ndimage.binary_dilation(fill_mat, structure=square(3))
    front = np.bitwise_xor(dilated, fill_mat)
    acts = np.where(front)
    return acts, dilated

def unravel_coordinate(node_index, fill_mat):
    coord = np.unravel_index(node_index, fill_mat.shape)
    return coord

def ravel_index(coord, fill_mat):
    index = np.ravel_multi_index(coord, fill_mat.shape)
    return index

def front_pressure(acts, dilated, Vi = 1, pc0=pc0, binary_flag=True):
    laplace = sp.ndimage.laplace(dilated)
    if binary_flag:
        shape_factor = -laplace[acts]-1
        shape_factor = Vi*shape_factor
        shape_factor[shape_factor<0] = 0
        pc = pc0*(1+shape_factor)
        # pc = pc0*(1 - 1/laplace[acts])
        # pc = pc0#*(-laplace[acts])
    else:
        pc = pc0*(2-laplace[acts])
    return pc

def get_node_gravity(node_index, fill_mat, V=1, rho=rho, g=g, grid_size=grid_size):
    coord = unravel_coordinate(node_index, fill_mat)
    y = (coord[0]-1+V)*grid_size
    pg = rho*g*y
    return pg

def init_K(acts, fills, K0, adj_matrix, K, Vi):
    # K = K_full.copy()
    K[:] = 0
    K[fills] = K0
    K[acts] = K0/Vi[acts]
    K_mat = 1/(1/K + 1/K[:,None])
    
    for i in acts:
        for j in acts:
            K_mat[i,j] = 0
    
    K_mat = K_mat*(adj_matrix)
    return K_mat
#  initialize

fill_mat = np.zeros(domain_size, dtype = np.bool)
fill_mat[:2,:] = True
# fill_mat[:58,10] = True
V_mat = np.zeros(domain_size, dtype = np.float32)

result_t_size = 800
result_array = np.zeros((domain_size[0], domain_size[1], result_t_size), dtype = np.bool)
result_pressure = np.zeros((domain_size[0], domain_size[1], result_t_size), dtype = np.float32)
result_time = np.zeros(result_t_size)
result_V = result_time.copy()
result_V[:] =np.nan

inlets = np.arange(domain_size[1])

pg = get_node_gravity(np.arange(domain_size[0]*domain_size[1]), fill_mat)

noise = 0.05
pc = pc0*(np.ones(len(pg))+noise*(-0.5+np.random.rand(len(pg))))
pc0 = pc.copy()
p = np.zeros(len(pg))

graph = nx.grid_2d_graph(domain_size[0], domain_size[1])
adj_matrix = nx.to_numpy_array(graph)

Vi = np.zeros(len(pg))
mask = np.zeros(len(pg), dtype=np.bool)
filled = mask.copy()
filled[ravel_index(np.where(fill_mat), fill_mat)] = True

Vi[filled] = 1
V_mat[fill_mat] = 1
mask[filled] = True

acts, dilated = front_extraction(fill_mat)
act_ind = ravel_index(acts, fill_mat)
mask[act_ind] = True

dt = 100000
time = 0

K = np.zeros(len(pg))

last_iteration = -1
ti = 0
for t in range(result_t_size*10): 
    if len(act_ind) == 0:
        last_iteration = ti
        break
    # K_mat = K_mat 
    # pc[act_ind] = front_pressure(acts, V_mat, pc0=pc0[act_ind], binary_flag = False)
    
    # 
    pc[act_ind] = front_pressure(acts, dilated*1, Vi=Vi[act_ind], pc0=pc0[act_ind])
    pg[act_ind] = get_node_gravity(act_ind, fill_mat, V=Vi[act_ind])
    
    K_mat = init_K(act_ind, filled, K0, adj_matrix, K, Vi)
    q_i, p = solve_pressure_field(p, mask, act_ind, inlets, K_mat, pg, pc)

    dt = 0.05/q_i[act_ind].max()
    if not np.any(dt*q_i>0.0001):
        last_iteration = ti
        break    
    if dt<0:
        dt = np.abs(dt)
    
    Vi[act_ind] = Vi[act_ind]  + dt*q_i[act_ind] #TODO get correct element size matching with conductivity, resolution etc
    filled[Vi>0.98] = True
    Vi[Vi<0] = 0
    V_mat = Vi.reshape(V_mat.shape)
    time = time + dt
    if np.any(filled[act_ind]):
        new_filled = act_ind[filled[act_ind]]
        fill_mat[unravel_coordinate(new_filled, fill_mat)] = True
        acts, dilated = front_extraction(fill_mat)
        act_ind = ravel_index(acts, fill_mat)
        mask[act_ind] = True

    
    if t%10 == 0:
        # print(q_i[act_ind].max())
        print(pc[act_ind])
        # print(unravel_coordinate(act_ind, fill_mat))
        # print(np.array(graph.nodes)[act_ind])
        result_array[:,:,ti] = fill_mat
        result_pressure[:,:,ti] = p.reshape(fill_mat.shape)
        result_time[ti] = time
        result_V[ti] = Vi.sum()
        ti = ti+1
    
# # TODO:maybe  add diagonal links with weight 1/sqrt(2)

