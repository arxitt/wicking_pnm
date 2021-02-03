# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:22:00 2021

@author: firo
"""


import sys

homeCodePath=r"H:\10_Python\005_Scripts_from_others\Laurent\wicking_pnm"
if homeCodePath not in sys.path:
	sys.path.append(homeCodePath)
    
    
import scipy as sp
import numpy as np
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import networkx as nx
from wickingpnm.model_for_art_sim import PNM
from wickingpnm.old_school_DPNM_v5 import simulation
import xarray as xr
import os
import robpylib
import pickle
import time

time0 = time.time()

xs = 2
ys = 2
zs = 3
n = xs*ys*zs

ecdf = robpylib.CommonFunctions.Tools.weighted_ecdf


sourceFolder = r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep"

#  extract distribution of peaks per pore
peak_num = np.array([])
comb_diff_data = np.array([])
comb_weight_data = np.array([])
samples = []
for sample in os.listdir(sourceFolder):
    if sample[:3] == 'dyn':
        data = xr.load_dataset(os.path.join(sourceFolder, sample))
        num_sample = data['dynamics'].sel(parameter = 'num filling peaks').data
        peak_num = np.concatenate([peak_num, num_sample])
        samples.append(data.attrs['name'])
    if sample[:14] == 'peak_diff_data':
        diff_data = xr.load_dataset(os.path.join(sourceFolder, sample))
        inter_diffs = diff_data['diffs_v2'][2:,:].data
        inter_weights = np.ones(inter_diffs.shape)
    
        intra_diffs = diff_data['diffs_v4'][2:,:].data
        intra_weights = np.ones(intra_diffs.shape) * diff_data['diffs_v4'][1,:].data
        intra_weights = 1- intra_weights
        
        diffs = np.concatenate([inter_diffs.flatten(), intra_diffs.flatten()], axis=0)
        weights = np.concatenate([inter_weights.flatten(), intra_weights.flatten()])
        
        comb_diff_data = np.concatenate([comb_diff_data, diffs], axis=0)
        comb_weight_data = np.concatenate([comb_weight_data, diffs], axis=0)
        
        
peak_num = peak_num[peak_num>1]

x, y = ecdf(peak_num)

peak_fun = interp1d(y, x, fill_value = 'extrapolate')


def generalized_gamma_cdf(x, xm, d, b, x0):
    y = sp.special.gammainc(d/b, ((x-x0)/xm)**b)/sp.special.gamma(d/b)
    return y

def weighted_ecdf(data, weight = False):
    """
    input: 1D arrays of data and corresponding weights
    sets weight to 1 if no weights given (= "normal" ecdf, but better than the statsmodels version)
    """
    if not np.any(weight):
        weight = np.ones(len(data))
    
    sorting = np.argsort(data)
    x = data[sorting]
    weight = weight[sorting]
    y = np.cumsum(weight)/weight.sum()
     
    # clean duplicates, statsmodels does not do this, but it is necessary for us
    
    x_clean = np.unique(x)
    y_clean = np.zeros(x_clean.shape)
    
    for i in range(len(x_clean)):
        y_clean[i] = y[x==x_clean[i]].max()
    return x_clean, y_clean

def from_ecdf(diff_data, n, seed=1):
    """
    
    Parameters
    ----------
    diff_data : netcdf4
        dataset containing waiting times as peak differences.
    n : int
        number of nodes.

    Returns
    -------
    array of waiting times with lentgh n.

    """
    inter_diffs = diff_data['diffs_v2'][2:,:].data
    inter_weights = np.ones(inter_diffs.shape)

    intra_diffs = diff_data['diffs_v4'][2:,:].data
    intra_weights = np.ones(intra_diffs.shape) * diff_data['diffs_v4'][1,:].data
    intra_weights = 1 - intra_weights
    
    diffs = np.concatenate([inter_diffs.flatten(), intra_diffs.flatten()], axis=0)
    weights = np.concatenate([inter_weights.flatten(), intra_weights.flatten()])

    mask = diffs>0

    x_t, y_t = weighted_ecdf(diffs[mask].flatten(), weights[mask].flatten())
    
    func = interp1d(y_t, x_t, fill_value = 'extrapolate')
    prng2 = np.random.RandomState(seed)
    waiting_times = func(prng2.rand(n))
    
    return waiting_times



def extend_waiting_time(waiting_times, waiting_time_gen, peak_num, diff_data, seed):
    size = len(waiting_times)
    for i in range(size):
        j = 1
        while j < peak_num[i]:
            j = j + 1
            waiting_times[i] = waiting_times[i] + waiting_time_gen(diff_data, 1, seed=j*i**2)[0]
    return waiting_times

def core_simulation(r_i, lengths, adj_matrix, inlets, timesteps,  pnm_params, peak_fun, i, pnm, diff_data, R0=1):
    size = len(r_i)
    prng = np.random.RandomState(i)
    waiting_times = from_ecdf(diff_data, size, seed=i+1)
    waiting_times = extend_waiting_time(waiting_times, from_ecdf, peak_fun(prng.rand(size)), diff_data, i)
    
    #  pass the pnm with the experimental activation time in the case of running the validation samples
    # time, V, V0, activation_time, filling_time = simulation(r_i, lengths, waiting_times, adj_matrix, inlets, timesteps, node_dict = pnm.label_dict, pnm = pnm, R0=R0,sample=pnm.sample)
    time, V, V0, activation_time, filling_time = simulation(r_i, lengths, waiting_times*0, adj_matrix, inlets, timesteps, node_dict = pnm.label_dict, R0=R0,sample=pnm.sample)
    V_fun = interp1d(time, V, fill_value = 'extrapolate')
    max_time = filling_time.max()
    # new_time = np.arange(3000)
    new_time = np.linspace(0,max_time,num=3000)
    new_V = V_fun(new_time)
    new_V[new_V>V0] = V0
   
    return new_time, new_V, V0, activation_time, filling_time, waiting_times

def core_function(samples, timesteps, i, peak_fun=peak_fun, inlet_count = 2, diff_data=None, xs=xs, ys=ys, zs=zs):
    prng3 = np.random.RandomState(i)
    sample = prng3.choice(samples)
    pnm_params = {
           'data_path': r"A:\Robert_TOMCAT_3_netcdf4_archives\for_PNM",
          # 'data_path': r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep",
            'sample': sample,
            # 'graph': nx.watts_strogatz_graph(n,8,0.1, seed=i+1),
            'graph': nx.convert_node_labels_to_integers(nx.grid_graph([xs,ys,zs])),
        # 'sample': 'T3_100_7_III',
        # 'sample': 'T3_025_3_III',
        # 'sample': 'T3_300_8_III',
          'inlet_count': inlet_count+2,
           'randomize_pore_data': True,
          'seed': (i+3)**3
    }

    pnm = PNM(**pnm_params)
    graph = pnm.graph.copy()
    
    # FIXME: change R to increase influence of pore filling and decrease effect of waiting
    R0 = 1E17#4E15
    
    # inlets = pnm.inlets.copy()
    inlets = np.arange(xs*ys)
    
    found_inlets = []
    for inlet in inlets:
        
        found_inlets.append(pnm.nodes[inlet])


    v_inlets = -1*(np.arange(len(inlets))+1)
    for k in range(len(inlets)):
        graph.add_edge(found_inlets[k], v_inlets[k])
    inlets = v_inlets
    
    inlet_radii = np.zeros(len(inlets))
    inlet_heights = np.zeros(len(inlets))  
    
    # TO DO: if you choose r_i and volume indepently from distribution, it is possible
    # to get infitive large pores --> change to aspect ratio or directly choose lengths
    r_i = pnm.radi.copy()
    # lengths = pnm.volumes.copy()/np.pi/r_i**2
    lengths = pnm.heights.copy()
    lengths[:] = lengths.mean()
    r_i[:] = r_i.mean()
    
    r_i = np.concatenate([r_i, inlet_radii])
    lengths = np.concatenate([lengths, inlet_heights])
    adj_matrix = nx.to_numpy_array(graph)
    
    if diff_data is None:    
        diff_data = pnm.pore_diff_data
    
    result_sim = core_simulation(r_i, lengths, adj_matrix, inlets, timesteps,  pnm_params, peak_fun, i, pnm, diff_data, R0=R0)
    
    # V0 = result_sim[2]
    time = result_sim[0]
    V = result_sim[1]
    ref = np.argmax(V)
    mean_flux = V[ref]/time[ref]
    result_sim = result_sim + (mean_flux,)
    result_sim = result_sim + (lengths,)
    result_sim = result_sim + (r_i,)
    result_sim = result_sim + (graph, )
    
    return result_sim

njobs = 32
timesteps = int(5000000*n/120)#0#0#0

# multi-sample run
paper_samples = [
    'T3_025_3_III',
    'T3_025_4',
    'T3_025_7_II',
    'T3_025_9_III',
    'T3_100_1',
    'T3_100_10',
    'T3_100_10_III',
    'T3_100_4_III',
    'T3_100_6',
    'T3_100_7',
    'T3_100_7_III',
    'T3_300_3',
    'T3_300_4',
    'T3_300_8_III',
    'T3_300_9_III'
]

not_extreme_samples = paper_samples
not_extreme_samples.remove('T3_100_1') #processing artefacts from moving sample
not_extreme_samples.remove('T3_025_4') #very little uptake --> v3
not_extreme_samples.remove('T3_025_9_III') #very little uptake --> v2,v3
# not_extreme_samples.remove('T3_300_4') #very little uptake
# not_extreme_samples.remove('T3_100_7') #very little uptake
temp_folder = None
temp_folder = r"Z:\users\firo\joblib_tmp"
# results = Parallel(n_jobs=njobs, temp_folder=temp_folder)(delayed(core_function)(not_extreme_samples, timesteps, i+5) for i in range(128))  
result = core_function(not_extreme_samples, timesteps, 5)
results = result
time_testing.append((n,time.time()-time0))

# dumpfilename = r"R:\Scratch\305\_Robert\simulation_dump\results_random2.p"
# dumpfile = open(dumpfilename, 'wb')
# pickle.dump(results, dumpfile)
# dumpfile.close()