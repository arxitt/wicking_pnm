# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:08:51 2020

@author: firo
"""

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt


# file = open(r"R:\Scratch\305\_Robert\simulation_dump\results.p",'rb')
# results = pickle.load(file)
# file.close()

# test: betweenness_centrality_subset
    # with inlets as sources
    #seems to make no difference, but try large number of runs
    
    # try to get the same comparison for the experiment
    # filling sequence from network and transition time


pearsons = np.zeros(len(results))
meanflux = pearsons.copy()
dragonwait = pearsons.copy()
dragonwait2 = pearsons.copy()
dragonwait3 = pearsons.copy()
dragonwait4 = pearsons.copy()
dragonwait5 = pearsons.copy()
dragonwait6 = pearsons.copy()
dragonwait7 = pearsons.copy()
dragonwait10 = pearsons.copy()
dragonwait20 = pearsons.copy()
dragonwait40 = pearsons.copy()
color = []
dragoncent = pearsons.copy()
dragoncent2 = pearsons.copy()
dragoncent3 = pearsons.copy()
dragoncent4 = pearsons.copy()
dragoncent5 = pearsons.copy()
dragoncent6 = pearsons.copy()
dragoncent7 = pearsons.copy()
dragoncent10 = pearsons.copy()
dragoncent20 = pearsons.copy()
dragoncent40 = pearsons.copy()
waitmean = pearsons.copy()
waitmedian = pearsons.copy()
weighted_wait = pearsons.copy()
weighted_wait2 = pearsons.copy()
weighted_wait3 = pearsons.copy()
weighted_wait4 = pearsons.copy()
weighted_wait5 = pearsons.copy()


cc = 0
for result in results:
    # sample = result[-1]
    
    col = 'k'
    if result[-4] == 100:
    # if sample[3:6] == '100':
        col = 'r'
    if result[-4] == 300:
    # if sample[3:6] == '300':
        col = 'b'
    color.append(col)
    
    waiting_times = result[5]
    # waiting_times = result[1]
    centrality = result[-3]
    centrality2 = result[-5]
    centrality3 = result[-6]
    centrality4 = result[-7]
    centrality5 = result[-8]

    centrality = centrality
    
    dg_size = 1#int(0.25*len(centrality))
    dragons = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 2#int(0.25*len(centrality))
    dragons2 = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 3#int(0.25*len(centrality))
    dragons3 = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 4#int(0.25*len(centrality))
    dragons4 = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 5#int(0.25*len(centrality))
    dragons5 = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 6#int(0.25*len(centrality))
    dragons6 = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 7#int(0.25*len(centrality))
    dragons7 = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 10#int(0.25*len(centrality))
    dragons10 = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 20#int(0.25*len(centrality))
    dragons20 = np.argpartition(centrality, -dg_size)[-dg_size:]
    dg_size = 40#int(0.25*len(centrality))
    dragons40 = np.argpartition(centrality, -dg_size)[-dg_size:]
    
    p = sp.stats.pearsonr(waiting_times, centrality)
    waitmean[cc] = np.mean(waiting_times)
    waitmedian[cc] = np.median(waiting_times)
    dragonwait[cc] = np.mean(waiting_times[dragons])
    dragonwait2[cc] = np.mean(waiting_times[dragons2])
    dragonwait3[cc] = np.mean(waiting_times[dragons3])
    dragonwait4[cc] = np.mean(waiting_times[dragons4])
    dragonwait5[cc] = np.mean(waiting_times[dragons5])
    dragonwait6[cc] = np.mean(waiting_times[dragons6])
    dragonwait7[cc] = np.mean(waiting_times[dragons7])
    dragonwait10[cc] = np.mean(waiting_times[dragons10])
    dragonwait20[cc] = np.mean(waiting_times[dragons20])
    dragonwait40[cc] = np.mean(waiting_times[dragons40])
    dragoncent[cc] = np.mean(centrality[dragons])
    dragoncent2[cc] = np.mean(centrality[dragons2])
    dragoncent2[cc] = np.mean(centrality[dragons2])
    dragoncent3[cc] = np.mean(centrality[dragons3])
    dragoncent4[cc] = np.mean(centrality[dragons4])
    dragoncent5[cc] = np.mean(centrality[dragons5])
    dragoncent6[cc] = np.mean(centrality[dragons6])
    dragoncent7[cc] = np.mean(centrality[dragons7])
    dragoncent10[cc] = np.mean(centrality[dragons10])
    dragoncent20[cc] = np.mean(centrality[dragons20])
    dragoncent40[cc] = np.mean(centrality[dragons40])
    
    weighted_wait[cc] = np.average(waiting_times, weights=centrality)
    weighted_wait2[cc] = np.average(waiting_times, weights=centrality**2)
    weighted_wait3[cc] = np.average(waiting_times, weights=centrality**3)
    weighted_wait4[cc] = np.sqrt(np.average(waiting_times**2, weights=centrality**2))
    weighted_wait5[cc] = np.sqrt(np.average(waiting_times**2, weights=centrality**3))
    
    pearsons[cc] = p[0]
    meanflux[cc] = result[-2]
    # meanflux[cc] = result[2]
    cc = cc+1

import matplotlib.colors as mcolors
import matplotlib.cm as cm
# normalize = mcolors.Normalize(vmin=meanflux.min(), vmax=meanflux.max())  
colormap = cm.plasma
 
# for i in range(len(meanflux)):
   
#     plt.plot(dragonwait3[i], meanflux[i], marker='.',color= colormap(normalize(dragoncent3[i])))
#     plt.text(dragonwait3[i], meanflux[i], str(i), color='k', fontsize=12)   
