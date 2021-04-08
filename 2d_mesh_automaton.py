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
pc = 1E5 #TODO: get reasonable value
g = 9.81 #m/s2
rho = 1000 #kg/m3
grid_size = 1E-3 #m

domain_size = (10, 10)

def solve_pressure_field(p, rest, masked, acts, inlets, K_mat, pg):
    # define RHS (boundary conditions)
    p[:]= pg  #TODO: add gravity
    p[acts] = pc[acts] + patm  #TODO: add perturbation and front shape effect
    p[inlets] = patm 
    
    # p = p + pg
    
    # define conductance (K_mat) and LHS (A) matrix

    A = - K_mat.copy()
    A[rest, :] = 0
    A[:, rest] = 0
    np.fill_diagonal(A, -A.sum(axis=0))
     
    A[inlets,:] = 0
    A[inlets,inlets] = 1
     
    for i in acts[0]:
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
    
    return q_i

# store filling state in numpy array v>=1:filled, 0:empty, 0<v<1 active
# dump this state to 3D array every once in a while -> result
# exctract actives and filled from this array and apply Laplace filter on it to get wetting force multiplicator at finger tips
#find good look up method for pore element coordinates from node indices (needed for gravity and update of filling array)
# include fiber orientation as non-isotropic weighting factor for wetting force
#  initializ with 2 filled rows, cinsider high resistzance between these rows to stabilize simulation
#  consider adding diagonal conductivity


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

def front_pressure(acts, dilated, pc0=pc):
    laplace = sp.ndimage.gaussian_laplace(dilated.astype(np.uint16), sigma=0.5)
    # laplace = sp.ndimage.laplace(dilated)
    pc = pc0*laplace[acts]
    return pc

def get_node_gravity(node_index, fill_mat, rho=rho, g=g, grid_size=grid_size):
    coord = unravel_coordinate(node_index, fill_mat)
    y = coord[1]*grid_size
    pg = rho*g*y
    return pg

#  initialize

fill_mat = np.zeros(domain_size, dtype = np.bool)
fill_mat[:2,:] = True
inlets = np.arange(domain_size[0])

pg = get_node_gravity(np.arange(domain_size[0]*domain_size[1]), fill_mat)
    
