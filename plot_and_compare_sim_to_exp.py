# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:55:51 2021

@author: firo
"""

import pickle
# import os
import numpy as np
import matplotlib.pyplot as plt


px = 2.75E-6
vx = px**3

results = pickle.load(open(r"R:\Scratch\305\_Robert\simulation_dump\results_random_wait_v3_not_calibrated_waiting_times.p", 'rb'))

results_sim = np.zeros((len(results),len(results[0][1])))

results_exp = np.load(r"H:\11_Essential_Data\03_TOMCAT\uptakes.npy")*vx

exp_mean = np.nanmean(results_exp, axis=0)
exp_std = np.nanstd(results_exp, axis=0)
exp_time = np.arange(exp_mean.size)
exp_min = np.nanmin(results_exp, axis=0)
exp_max = np.nanmax(results_exp, axis=0)


for i in range(len(results)):
    result = results[i]
    results_sim[i,:] = result[1]
    
sim_mean = np.nanmean(results_sim, axis=0)
sim_std = np.nanstd(results_sim, axis=0)
sim_time = np.arange(sim_mean.size)
sim_min = np.nanmin(results_sim, axis=0)
sim_max = np.nanmax(results_sim, axis=0)

plt.plot(exp_mean, 'k')
plt.fill_between(exp_time, exp_mean+exp_std, exp_mean-exp_std, color='k', alpha=0.2)
plt.plot(exp_min, 'k--')
plt.plot(exp_max, 'k--')

plt.plot(sim_mean, 'r')
plt.fill_between(sim_time, sim_mean+sim_std, sim_mean-sim_std, color='r', alpha=0.2)
plt.plot(sim_min, 'r--')
plt.plot(sim_max, 'r--')

plt.title('R=0.5E17, waiting_times calibrated')
# plt.title('R=1, waiting_times not calibrated')
# plt.title('R=0.5E17, waiting_times not calibrated')


plt.xlim(0,1600)
    