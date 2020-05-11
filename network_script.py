# -*- coding: utf-8 -*-
"""
Created on Wed May  6 08:19:34 2020

@author: firo
"""


import xarray as xr
import numpy as np
import scipy as sp
import scipy.sparse
# import os
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import robpylib
import networkx as nx

net_stat_path = r"W:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep\network_statistics.nc"
exp_data_path = r"W:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep\dyn_data_T3_025_3_III.nc"
pore_data_path = r"W:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep\pore_props_T3_025_3_III.nc"

#  define some general physical constants
eta = 1 #mPa*s dynamic viscosity of water
gamma = 72.6 #mN/m surface tension of water
theta = 50 #° contact angle
cos = np.cos(theta/180*np.pi)
px = 2.75E-6 #m

# pass some empirical information
inlets = [162, 171, 207]
R_inlet = 5E19 #Pas/m3

# intialize simulation boundaries
t_init = 1E-4 #s start time to stabilze simulation and avoid inertial regime, now irrelavent because flow rate is solved iteratively
tmax = 1600 #s
dt = 1E-4#s



#  load waiting time statistics
delta_t_025 = np.array([])
delta_t_100 = np.array([])
delta_t_300 = np.array([])

network_statistics = xr.load_dataset(net_stat_path)
for key in list(network_statistics.coords):
    if not key[-2:] == '_t': continue
    if key[3:6] == '025':
        delta_t_025 = np.concatenate([delta_t_025, network_statistics[key].data])
    if key[3:6] == '100':
        delta_t_100 = np.concatenate([delta_t_100, network_statistics[key].data])
    if key[3:6] == '300':
        delta_t_300 = np.concatenate([delta_t_300, network_statistics[key].data])
delta_t_all = network_statistics['deltatall'].data


# extract the network
data = xr.load_dataset(exp_data_path)
label_matrix = data['label_matrix'].data
labels = data['label'].data

adj_mat = robpylib.CommonFunctions.pore_network.adjacency_matrix(label_matrix)
# remove diagonal entries (self-loops)
adj_mat[np.where(np.diag(np.ones(adj_mat.shape[0], dtype=np.bool)))] = False

# remove irrelevant/noisy labels, pores that are just a few pixels large
mask = np.ones(adj_mat.shape[0], np.bool)
mask[labels] = False
adj_mat[mask,:] = False
adj_mat[:,mask] = False

# construct networkx graph object, not necessary but allows the use of random graphs in the simulation function
adj_sparse = sp.sparse.coo_matrix(adj_mat)
conn_list = zip(adj_sparse.row, adj_sparse.col)
expnet = nx.Graph()
expnet.add_edges_from(conn_list)

# load pore properties
pore_data = xr.load_dataset(pore_data_path)
re = np.sqrt(pore_data['value_properties'].sel(property = 'median_area').data/np.pi)*px
h0e = pore_data['value_properties'].sel(property = 'major_axis').data*px


# function to calculate the resistance of a full pore
def poiseuille_resistance(l, r, eta=eta):
    R = 8*eta*l/np.pi/r**4
    return R

# function to calculate the filling velocity considering the inlet resistance and tube radius
def capillary_rise(t, r, R0, cos = cos, gamma = gamma, eta =eta):
    dhdt = gamma*r*cos/2/eta/np.sqrt((R0*np.pi*r**4/8/eta)**2+gamma*r*cos*t/2/eta)
    return dhdt

# use capillary_rise2 because the former did not consider that R0 can change over time, should be irrelevant because pore contribution becomes quickly irrelevant , but still...
def capillary_rise2(r, R0, h, cos = cos, gamma = gamma, eta = eta):
    dhdt = 2*gamma*cos/(R0*np.pi*r**3+8*eta*h/r)
    return dhdt

#  wrap up pore filling states to get total amount of water in the network
def total_volume(h, r):
    V = (h*np.pi*r**2).sum()
    return V

#  simple
def effective_resistance(R_nb):
    R = 1/(1/R_nb).sum()
    return R

"""
find your path through the filled network to calculate the inlet
resistance imposed on the pores at the waterfront
quick and dirty, this part makes the code slow and might even be wrong
we have to check
"""
def outlet_resistances(inlets, filled, R_full, net):
    
    # initialize pore resistances
    R0 = np.zeros(len(filled))
    # initialize "to-do list", only filled pores contribute to the network permeability
    to_visit = np.zeros(len(filled), dtype=np.bool)
    to_visit[np.where(filled)] = True
    
    #  check if inlet pores are already filled
    filled_inlets = inlets.copy()
    for nd in filled_inlets:
        if filled[nd] == False:
            filled_inlets.remove(nd)
    
    # this part iteratively (should) calculate the effective inlet resistance
    # for every pore with the same distance (current_layer) to the network inlet
    current_layer = filled_inlets
    to_visit[filled_inlets] = False
    
    while True in to_visit:
        next_layer = []
        for nd in current_layer:
            next_layer = next_layer + list(net.neighbors(nd))
        next_layer = list(np.unique(next_layer))
        for nd in next_layer:
            nnbb = list(net.neighbors(nd))
            R_nb = []
            for nb in nnbb:
                if nb in current_layer:
                    R_nb.append(R0[nb]+R_full[nb])
            R0[nd] = effective_resistance(np.array(R_nb))
            to_visit[nd] = False
        current_layer = next_layer.copy()
        for nd in current_layer:
            if filled[nd] == False:
                current_layer.remove(nd)
    return R0

def simulation3(net = expnet, t_init=t_init, tmax=tmax, dt=dt, R_inlet=R_inlet, re = re, h0e = h0e, t_wait_seq = False, inlets = False):
   
    # this part is necessary to match the network pore labels to the pore property arrays
    n = len(net.nodes)  
    n_init = np.array(net.nodes).max()+1
    node_ids = list(net.nodes)
    node_ids.sort()
    node_ids = np.array(node_ids)
    
    
    num_inlets = max(int(0.1*n),6)
    if not np.any(inlets):
        inlets = np.random.choice(net.nodes, num_inlets)
        inlets = list(np.unique(inlets))
    temp_lets = []
    
    # double-check if inlet pores are actually in the network
    for inlet in inlets:
        if inlet in net:
            temp_lets.append(inlet)
    inlets = temp_lets
    # print(inlets)


    # asign a random waiting time to every pore based on the experimental distribution
    if np.any(t_wait_seq):       
        ecdf = ECDF(t_wait_seq)
        f = interp1d(ecdf.y[1:], ecdf.x[1:], fill_value = 'extrapolate')
        prob = np.random.rand(n_init)
        t_w = f(prob)     
    #t_w = t_w*0  


    # create new pore property arrays where the pore label corresponds to the array index
    # this copuld be solved more elegantly with xarray, but the intention was that it works
    
    time = np.arange(t_init, tmax, dt)
    act_time = np.zeros(n_init)
    filled = np.zeros(n_init, dtype = np.bool)
    h = np.zeros(n_init)+1E-6
    R0 = np.zeros(n_init)
    r = np.zeros(n_init)
    h0 = np.zeros(n_init)
    cc=0
    for node_id in node_ids:
        r[node_id] = re[cc]
        h0[node_id] =h0e[cc]
        cc=cc+1
    
    R0[inlets] = R_inlet
    V=np.zeros(len(time))
    R_full = poiseuille_resistance(h0, r) +R0
    
    
    # this is the simulation:
    active = inlets
    new_active = []
    
    # every time step solve explicitly
    tt=0
    for t in time:
        
        # first check, which pores are currently getting filled (active)
        new_active = list(np.unique(new_active))
        if len(new_active)>0:
            for node in new_active:
                if filled[node] == True:
                    new_active.remove(node)
            act_time[new_active] = t + t_w[new_active]
            active = active + new_active
            R0 = outlet_resistances(inlets, filled, R_full, net)
        active = list(np.unique(active))
        new_active = []
        
        # calculate the new filling state (h) for every active pore
        for node in active:
            if t>act_time[node]:
                h_old = h[node]
                #h[node] = h[node] + dt*capillary_rise(t-act_time[node], r[node], R0[node])
                if node in inlets:
                    #patch to consider inlet resitance at inlet pores
                    h[node] = h_old + dt*capillary_rise2(r[node], R0[node]+ R_inlet, h_old)
                else:
                    #influence of inlet resistance on downstream pores considered by initialization of poiseuille resistances
                    h[node] = h_old + dt*capillary_rise2(r[node], R0[node], h_old)
                
                # if pore is filled, look for neighbours that would now start to get filled
                if h[node] >= h0[node]:
                    h[node] = h0[node]
                    filled[node] = True
                    active.remove(node)
                    new_active = new_active + list(net.neighbors(node))              
                
        V[tt] = total_volume(h[node_ids], r[node_ids])
        tt=tt+1
    return(time, V)


# run the simulation 20 times in parallel
# results = Parallel(n_jobs = mp.cpu_count())(delayed(simulation3)(net=expnet, inlets = inlets, t_init=t_init, tmax=tmax, dt=dt, t_wait_seq = tseq, R_inlet=R_inlet) for i in range(20))

# or just once
results = []
result = simulation3(net=expnet, inlets = inlets, t_init=t_init, tmax=tmax, dt=dt, t_wait_seq = delta_t_025, R_inlet=R_inlet)
results.append(result)


#  and plot
for result in results:
    plt.loglog(result[0], result[1])   
plt.title('experimental network')
plt.xlabel('time [s]')
plt.ylabel('volume [m3]')
plt.xlim(0.1,tmax)

plt.figure()
for result in results:
    plt.plot(result[0], result[1])   
plt.title('experimental network')
plt.xlabel('time [s]')
plt.ylabel('volume [m3]')

plt.figure()
Qmax = 0
for result in results:
    Q = np.gradient(result[1], result[0])
    Qmax = np.max([Qmax, Q[5:].max()])
    plt.plot(result[0], np.gradient(result[1]))
    
plt.title('experimental network')
plt.xlabel('time [s]')
plt.ylabel('flux [m3/s]')
plt.ylim(0, Qmax)


## compare to experimental data
plt.figure()
vxm3 = px**3
test = np.array(results)
std = test[:,1,:].std(axis=0)
mean =test[:,1,:].mean(axis=0)
time_line = test[0,0,:]

time_coarse = time_line[::1000]
mean_coarse = mean[::1000]
std_coarse = std[::1000]

plt.plot(time_coarse, mean_coarse)#)
plt.fill_between(time_coarse, mean_coarse-std_coarse, mean_coarse+std_coarse, alpha=0.2)
(data['volume'].sum(axis = 0)*vxm3).plot(color='k')