# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:55:51 2021

@author: firo
"""

import pickle
# import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


px = 2.75E-6
vx = px**3

results = pickle.load(open(r"R:\Scratch\305\_Robert\simulation_dump\results_random_wait_v3_R0_no_extend_v2_512runs_repeat.p", 'rb'))

results_sim = np.zeros((len(results),len(results[0][1])))

# results_exp = np.load(r"H:\11_Essential_Data\03_TOMCAT\uptakes.npy")*vx
results_exp = np.load(r"H:\11_Essential_Data\03_TOMCAT\uptakes_combined_3_3bv2.npy")

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

plt.plot(exp_mean, 'r')
plt.fill_between(exp_time, exp_mean+exp_std, exp_mean-exp_std, color='r', alpha=0.2)
plt.plot(exp_min, 'r--')
plt.plot(exp_max, 'r--')

plt.plot(sim_mean, 'k')
plt.fill_between(sim_time, sim_mean+sim_std, sim_mean-sim_std, color='k', alpha=0.2)
plt.plot(sim_min, 'k--')
plt.plot(sim_max, 'k--')

# plt.title('R=4E17')
# plt.title('R=1, waiting_times not calibrated')
# plt.title('R=0.5E17, waiting_times not calibrated')
plt.xlabel('time [s]')
plt.ylabel('volume [m3]')

plt.xlim(0,1600)
plt.ylim(0,9E-11)

filepath = r"R:\Scratch\305\_Robert\pnm_to_exp_R4_v3_with3b.tex"
tikzplotlib.save(filepath)