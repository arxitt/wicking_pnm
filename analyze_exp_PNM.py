# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:23:07 2020

@author: firo
"""

import sys

homeCodePath=r"H:\10_Python\005_Scripts_from_others\Laurent\wicking_pnm"
if homeCodePath not in sys.path:
	sys.path.append(homeCodePath)


import numpy as np
from joblib import Parallel, delayed
from wickingpnm.model import PNM
import networkx as nx

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
# not_extreme_samples.remove('T3_100_1') #processing artefacts from moving sample
# not_extreme_samples.remove('T3_025_4') #very little uptake --> v3
not_extreme_samples.remove('T3_025_9_III') #very little uptake --> v2,v3

def function(sample):
    pnm_params = {
            'data_path': r"Z:\Robert_TOMCAT_3_netcdf4_archives\for_PNM",
           # 'data_path': r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep",
            'sample': sample
        # 'sample': 'T3_100_7_III',
        # 'sample': 'T3_025_3_III',
        # 'sample': 'T3_300_8_III',
    }
    pnm = PNM(**pnm_params)
    graph = pnm.graph
    sourcesraw = np.unique(pnm.data['label_matrix'][:,:,0])[1:]
    targetsraw = np.unique(pnm.data['label_matrix'][:,:,-1])[1:]
    sources = []
    targets = []
    for k in sourcesraw:
        if k in graph.nodes():
            sources.append(k)
    for k in targetsraw:
        if k in graph.nodes():
            targets.append(k)
    centrality = np.array(list(nx.betweenness_centrality(graph).values()))
    centrality2 = np.array(list(nx.betweenness_centrality_subset(graph, sources, targets, normalized=True).values()))
    centrality5 = np.array(list(nx.betweenness_centrality_source(graph, sources=sources).values()))
    
    edge_centrality = np.array(list(nx.edge_betweenness_centrality(graph).values()))
    edge_waiting = np.zeros(len(edge_centrality))
    
    waiting_times = np.zeros(len(centrality))
    vx = pnm.data.attrs['voxel']
    V = pnm.data['volume'].sum(dim='label')
    V0 = 0.95*V.max()
    ref = V[V<V0].argmax()
    meanflux = (V0*vx**3/pnm.data['time'][ref])
    # meanflux = V[-1]*vx**3/pnm.data['time'][-1]
    meanflux = meanflux.data
    
    i = 0
    for node in graph.nodes():
        wt = 0
        nb = list(nx.neighbors(graph, node))
        if len(nb)>0:
            t0 = pnm.data['sig_fit_data'].sel(label = node, sig_fit_var = 't0 [s]')
            t0nb = pnm.data['sig_fit_data'].sel(label = nb, sig_fit_var = 't0 [s]')
            upstream_nb = np.where(t0nb < t0)
            if len(upstream_nb[0]) > 0:
                wt = t0 - t0nb[upstream_nb].max()
        waiting_times[i] = wt
        i += 1
    ii = 0   
    for edge in graph.edges():
        wt = 0
        t0 = pnm.data['sig_fit_data'].sel(label = edge[0], sig_fit_var = 't0 [s]')
        t0nb = pnm.data['sig_fit_data'].sel(label = edge[1], sig_fit_var = 't0 [s]')
        wt = np.abs(t0nb-t0)
        edge_waiting[ii] = wt
        ii += 1
    
    return centrality, waiting_times, meanflux,centrality5, centrality2, sample, edge_centrality, edge_waiting

njobs = 16
# temp_folder = r"Z:\users\firo\joblib_tmp"
temp_folder = None
results = Parallel(n_jobs=njobs, temp_folder=temp_folder)(delayed(function)(sample) for sample in not_extreme_samples)

meanfluxes = np.zeros(len(results))
meancentrality = meanfluxes.copy()
weighted_edge_wait = meanfluxes.copy()
mean_wait = meanfluxes.copy()

cc= 0
for result in results:
    meanfluxes[cc] = result[2]
    edge_centrality = result[-2]
    edge_wait = result[-1]
    meancentrality[cc] = edge_centrality.mean()
    weighted_edge_wait[cc] = np.average(edge_wait, weights=edge_centrality**3)
    mean_wait[cc] = edge_wait.mean()
    cc += 1