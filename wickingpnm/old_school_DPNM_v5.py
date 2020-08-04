# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:50:08 2020

@author: firo
"""

import numpy as np
import networkx as nx

# parameters


eta = 1E-3 #Pas
gamma = 72E-3 #N/m
n = 4
a = 2
cos = np.cos(48/180*np.pi)

timesteps = 100000
patm = 101.325E3 #Pa

dim = [n,n,a*n]


def active_K(h, r, eta=eta):
    return np.pi*r**4/8/h/eta

def timestep(q_i, r_i, h, l, dxmax = 0.5E-4):
    
    hmax = np.abs((l-h).min())
    if hmax < dxmax:
        dxmax = hmax
        rmin = r_i[q_i>0].min()
        
        if dxmax < rmin:
            dxmax = rmin
    dt = (dxmax*np.pi*r_i[q_i>0]**2/(q_i[q_i>0])).min()
    return dt
     
def init_K(acts, fills, r_i, K_full, adj_matrix, K, heights):
    # K = K_full.copy()
    K[:] = 0
    K[fills] = K_full[fills].copy()/2
    K[acts] = active_K(heights[acts]+1E-6, r_i[acts])/2
    
    K_mat = 1/(1/K + 1/K[:,None])
    
    for i in acts[0]:
        for j in acts[0]:
            K_mat[i,j] = 0
    
    K_mat = K_mat*(adj_matrix)
    return K_mat
 
# create network

def init_regular_grid(dim):
    graph = None
    graph = nx.grid_graph(dim)
    graph = nx.convert_node_labels_to_integers(graph)
    size = len(graph.nodes)
    
    adj_matrix = nx.to_numpy_array(graph)
    
    # r_i = np.random.rand(size)*1E-5
    # lengths = np.random.rand(size)*1E-2
    
    r_i = np.ones(size)*1E-5# + np.random.rand(size)*1E-5
    lengths = np.ones(size)*1E-2# + np.random.rand(size)*1E-2
    waiting_times = np.random.rand(size)*0
    inlets = np.arange(dim[0]*dim[1])
    
    return adj_matrix, r_i, lengths, waiting_times, inlets


def simulation(r_i, lengths, waiting_times, adj_matrix, inlets, timesteps, patm = patm, eta = eta, gamma = gamma, cos = cos):
    size = adj_matrix.shape[0]
    
    # initialize arrays
    activation_time = np.zeros(size)
    filling_time = np.zeros(size)
    heights = np.zeros(size)
    K = np.zeros(size)
    p = np.zeros(size)
    
    filled = np.zeros(size, dtype=int)
    active = filled.copy()
    mask = np.zeros(size, dtype=int)
    
    V = np.zeros(timesteps)
    time = V.copy()


    # get static pore properties
    K_full = np.pi*r_i**4/8/eta/lengths
    pc = 2*gamma*cos/r_i 
    
    
    # initialize network state
    filled[inlets] = 1
    fills = np.where(filled)
    active[np.unique(np.where(adj_matrix[fills,:]))]=1
    active[fills]=0
    heights[fills] = lengths[fills]
    acts = np.where(active)
    V0 = (lengths*np.pi*r_i**2).sum()
    dt=0
    
    # run simulation
    for t in range(timesteps):
        
        # let loop roll off if full saturation is reached
        if V[t-1] >= V0: 
            time[t] = time[t-1] + dt
            V[t] = V[t-1]
            continue
        
        # TODO: add condition if all active pores are waiting and increment time step if so
        
        old_heights = heights.copy()
        
        # select filled sub-network behind the waterfront
        mask[:] = 0
        mask[inlets] = 1
        mask[fills] = 1
        mask[acts] = 1
        masked = np.where(mask>0)[0]
        act_waiting = np.where(activation_time>time[t-1])
               
        # define RHS (boundary conditions)
        p[:]= 0
        p[acts] = pc[acts] + patm
        p[act_waiting] = 0
        p[inlets] = patm

        # define conductance (K_mat) and LHS (A) matrix
        K_mat = init_K(acts, fills, r_i, K_full, adj_matrix, K, heights)
        A = - K_mat.copy()
        np.fill_diagonal(A, -A.sum(axis=0))
         
        A[inlets,:] = 0
        A[inlets,inlets] = 1
            
        for i in acts[0]:
            if i in act_waiting[0]: continue
            A[i,:] = 0
            A[i,i] = 1  
        
        A = A[masked,:]
        A = A[:,masked] 
  
        # solve equation system for pressure field
        p[masked] = np.linalg.solve(A, p[masked])            
        p_mat = p-p[:,None]
      
       # get pore fluxes
        q_ij = K_mat*p_mat
        q_i = q_ij.sum(axis=0)
    
       # get new time stepping
        if np.any(q_i>0):
            dt = 1.0001*timestep(q_i, r_i, old_heights[acts], lengths[acts])
          
    #  update pore filling states
        heights = heights + dt*q_i/np.pi/r_i**2
                
        old_filled = filled.copy()
        filled[heights>=lengths] = 1
        fills = np.where(filled)
        
        heights[heights<0] = 0
        heights[fills] = lengths[fills]
        
        active[:] = 0
        active[np.unique(np.where(adj_matrix[fills,:]))] = 1
        active[fills]=0
        
        new_actives = np.where((active>0)*~(activation_time>0))
        activation_time[new_actives] = waiting_times[new_actives] + time[t-1]
        
        acts = np.where(active>0)
        
        # wrap up results
        time[t] = time[t-1] + dt
        filling_time[np.where(filled-old_filled>0)] = time[t]
        V[t] = (heights*np.pi*r_i**2).sum()
        
    return time, V, V0, activation_time, filling_time

#adj_matrix, r_i, lengths, waiting_times, inlets = init_regular_grid(dim)  
#time, V, V0, activation_time, filling_time = simulation(r_i, lengths, waiting_times, adj_matrix, inlets, timesteps)
