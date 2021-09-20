# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:50:08 2020

@author: firo
"""

import numpy as np
# import time as systime
# parameters

use_gpu = True
if use_gpu:
    import cupy as cp
else:
    cp = np
    

eta = 1E-3 #Pas
gamma = 72E-3 #N/m
n = 6
a = 2
cos = cp.cos(48/180*cp.pi)
R0 = 1

timesteps = 100000
patm = 101.325E3 #Pa

dim = [n,n,a*n]



def active_K(h, r, eta=eta):
    return cp.pi*r**4/8/h/eta

def timestep(q_i, r_i, h, l, dxmax = 0.5E-4):
    
    hmax = cp.abs((l-h).min())
    if hmax < dxmax:
        dxmax = hmax
        rmin = r_i[q_i>0].min()
        
        if dxmax < rmin:
            dxmax = rmin
    dt = (dxmax*cp.pi*r_i[q_i>0]**2/(q_i[q_i>0])).min()
    dt = max((dt,0.005))
    return dt
     
def init_K(acts, fills, r_i, K_full, adj_matrix, K, heights):
    # K = K_full.copy()
    K[:] = 0
    K[fills] = K_full[fills].copy()#/2
    K[acts] = active_K(heights[acts]+1E-6, r_i[acts])#/2  #comparing to Washburn suggests to not divide by two, but 4 ?!
    
    K_mat = 1/(1/K + 1/K[:,None])
    
    for i in acts[0]:
        for j in acts[0]:
            K_mat[i,j] = 0
    
    K_mat = K_mat*(adj_matrix)
    return K_mat
 
# create network

def simulation(r_i, lengths, waiting_times, adj_matrix, inlets,  timesteps, sig_diffs = None, node_dict = None, patm = patm, eta = eta, gamma = gamma, cos = cos, R0 = R0, pnm=None, sample=None, tlim=1E16):
    
    size = adj_matrix.shape[0]
    
    # initialize arrays
    activation_time = cp.zeros(size)
    filling_time = cp.zeros(size)
    heights = cp.zeros(size)
    K = cp.zeros(size)
    p = cp.zeros(size)
    q_i = cp.zeros(size)
    
    filled = cp.zeros(size, dtype=int)
    active = filled.copy()
    mask = cp.zeros(size, dtype=int)
    
    V = cp.zeros(timesteps)
    time = V.copy()


    # get static pore properties
    K_full = cp.pi*r_i**4/8/eta/lengths
    K_full[inlets] = 1/R0
    pc = 2*gamma*cos/r_i 
    
    
    filled[inlets] = 1
    # fills = cp.where(filled)
    fills = filled.nonzero()[0]
    # active[cp.unique(cp.where(adj_matrix[fills,:]))]=1
    active[adj_matrix[fills,:].nonzero()[0]] = 1
    active[fills] = 0
    heights[fills] = lengths[fills]
    acts = cp.where(active)
    V0 = (lengths*cp.pi*r_i**2).sum()
    dt=0.005
    
    # run simulation
    for t in range(timesteps):
        # flag = False
        # if t % 5000 == 0: flag=True
        # let loop roll off if full saturation is reached
        if V[t-1] >= V0 or time[t-1]>tlim: 
            time[t] = time[t-1] + dt
            V[t] = V[t-1]
            continue
        
       
        old_heights = heights.copy()
        
        # select filled sub-network behind the waterfront
        act_waiting = cp.where(activation_time>time[t-1])
        mask[:] = 0
        mask[inlets] = 1
        mask[fills] = 1
        mask[acts] = 1
        mask[act_waiting] = 0
        masked = cp.where(mask>0)[0]
        rest = cp.where(mask==0)[0]
        
        #  fast-foward if there is no active pore
        if not np.any(mask[acts] > 0):
            time[t] = activation_time[activation_time>time[t-1]].min()
            
            V[t] = V[t-1]
            continue 
    
               
        # define RHS (boundary conditions)
        p[:]= 0
        p[acts] = pc[acts] + patm
        p[act_waiting] = 0
        p[inlets] = patm #np.min([patm + np.abs(R0*q_i[inlets]), pc[inlets] + patm], axis = 0)
        
        
        # define conductance (K_mat) and LHS (A) matrix
        K_mat = init_K(acts, fills, r_i, K_full, adj_matrix, K, heights)
        A = - K_mat.copy()
        A[rest, :] = 0
        A[:, rest] = 0
        cp.fill_diagonal(A, -A.sum(axis=0))
        
        A[inlets,:] = 0
        A[inlets,inlets] = 1
            
        for i in acts[0]:
            A[i,:] = 0
            A[i,i] = 1  
        
        A = A[masked,:]
        A = A[:,masked] 
  
        # solve equation system for pressure field
        p[masked] = cp.linalg.solve(A, p[masked])            
        p_mat = p-p[:,None]
      
       # get pore fluxes
        q_ij = K_mat*p_mat
        q_i = q_ij.sum(axis=0)
    
       # get new time stepping
        if cp.any(q_i>0):
            dt = 1.0001*timestep(q_i, r_i, old_heights[acts], lengths[acts])

        
    #  update pore filling states
        heights = heights + dt*q_i/cp.pi/r_i**2
        # heights[acts] = heights[acts] + dt*q_i[acts]/np.pi/r_i[acts]**2
        old_filled = filled.copy()
        filled[heights>=lengths] = 1
        
        #  allow draining of pores
        # filled[heights<lengths]  = 0
        # fills = cp.where(filled)
        fills = filled.nonzero()[0]
        
        heights[heights<0] = 0
        heights[fills] = lengths[fills]
        
        active[:] = 0
        # active[cp.unique(cp.where(adj_matrix[fills,:]))] = 1
        active[adj_matrix[fills,:].nonzero()[0]]
        active[fills] = 0
        
        new_actives = cp.where((active>0)*~(activation_time>0))
        
        activation_time[new_actives] = waiting_times[new_actives] + time[t-1]
    #     if pnm is not None:
    #         for n in new_actives[0]:
    #             if pnm.nodes[n] in pnm.data['label']:
    #                 texp = pnm.data['sig_fit_data'].sel(sig_fit_var = 't0 [s]', label= pnm.nodes[n])
    #                 activation_time[n] = texp


    #             if sample == 'T3_025_3_III':
    # #                   late fills
    #                 if pnm.nodes[n] == 25:
    #                       activation_time[n] = time[t-1] + waiting_times[n]
    #                 if pnm.nodes[n] == 31:
    #                       activation_time[n] = time[t-1] + waiting_times[n]
    #                 if pnm.nodes[n] == 149:
    #                     activation_time[n] = time[t-1] + waiting_times[n]
    #                 if pnm.nodes[n] == 69:
    #                     activation_time[n] = time[t-1] + waiting_times[n]
    #                 if pnm.nodes[n] == 24:
    #                     activation_time[n] = time[t-1] + waiting_times[n]
    #                     # [162,171,207]
    #                     # inlets
    #                 if pnm.nodes[n] == 162:
    #                     activation_time[n] = 200
    #                 if pnm.nodes[n] == 207:
    #                     activation_time[n] = 200   
    #                 if pnm.nodes[n] == 171:
    #                     activation_time[n] = 200    
                        
    #             if sample == 'T3_100_7_III':
    #                 # inlets [86, 89, 90]
    #                 if pnm.nodes[n] == 86:
    #                     activation_time[n] = 31
    #                 if pnm.nodes[n] == 89:
    #                     activation_time[n] = 20
    #                 if pnm.nodes[n] == 90:
    #                     activation_time[n] = 20                
    #                 if pnm.nodes[n] == 52:
    #                     activation_time[n] = 150                      
            
                    
        acts = cp.where(active>0)
        
        # wrap up results
        time[t] = time[t-1] + dt
        filling_time[cp.where(filled-old_filled>0)] = time[t]
        V[t] = (heights*cp.pi*r_i**2).sum()
        
    return time, V, V0, activation_time, filling_time

#  multiply waiting time with distribution of peaks in pore, 
# better idea: add probable number of waiting intervals (less prone to outlieres)
# do this! it's a safe bet

# adj_matrix, r_i, lengths, waiting_times, inlets = init_regular_grid(dim)  
# time, V, V0, activation_time, filling_time = simulation(r_i, lengths, waiting_times, adj_matrix, inlets, timesteps)