#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 08:19:34 2020

@author: firo
"""

import sys
import argparse

import xarray as xr
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing as mp

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from skimage.morphology import cube
from collections import deque

job_count = 4 # Default job count, 4 should be fine on most systems

## define some general physical constants
eta = 1 #mPa*s dynamic viscosity of water
gamma = 72.6 #mN/m surface tension of water
theta = 50 #° contact angle
cos = np.cos(theta/180*np.pi)
px = 2.75E-6 #m

## function to calculate the resistance of a full pore
poiseuille_resistance = lambda l, r, eta=eta: \
    8*eta*l/np.pi/r**4

## function to calculate the filling velocity considering the inlet resistance and tube radius
capillary_rise = lambda t, r, R0, cos = cos, gamma = gamma, eta = eta: \
    gamma*r*cos/2/eta/np.sqrt((R0*np.pi*r**4/8/eta)**2+gamma*r*cos*t/2/eta)

## use capillary_rise2 because the former did not consider that R0 can change over time, should be irrelevant because pore contribution becomes quickly irrelevant , but still...
capillary_rise2 = lambda r, R0, h, cos = cos, gamma = gamma, eta = eta: \
    2*gamma*cos/(R0*np.pi*r**3+8*eta*h/r)

## wrap up pore filling states to get total amount of water in the network
total_volume = lambda h, r: \
    (h*np.pi*r**2).sum()

##  simple
effective_resistance = lambda R_nb: \
    1/(1/R_nb).sum()

class WickingPNMStats:
    def __init__(self, path):
        # waiting time statistics
        self.delta_t_025 = np.array([])
        self.delta_t_100 = np.array([])
        self.delta_t_300 = np.array([])
        self.delta_t_all = np.array([])

        print('Reading the statistics dataset at {}'.format(path))
        stats_dataset = xr.load_dataset(path)

        for key in list(stats_dataset.coords):
            if not key[-2:] == '_t': continue
            if key[3:6] == '025':
                self.delta_t_025 = np.concatenate([self.delta_t_025, stats_dataset[key].data])
            if key[3:6] == '100':
                self.delta_t_100 = np.concatenate([self.delta_t_100, stats_dataset[key].data])
            if key[3:6] == '300':
                self.delta_t_300 = np.concatenate([self.delta_t_300, stats_dataset[key].data])

        self.delta_t_all = stats_dataset['deltatall'].data

class WickingPNM:
    def __init__(self, generate, exp_data_path = None, pore_data_path = None, stats_path = None):
        self.data = None
        self.graph = None
        self.waiting_times = np.array([])
        self.params = {
            ## intialize simulation boundaries
            # t_init is now irrelevent because flow rate is solved iteratively
            't_init': 1E-4, # (seconds) start time to stabilze simulation and avoid inertial regime
            'tmax': 1600, # (seconds)
            'dt': 1E-4, # (seconds)

            # TODO: Describe the below
            'R_inlet': 0,
            're': 0,
            'h0e': 0,
            'inlets': False,
        }

        if not generate:
            print('Generating the network from data')
            stats = WickingPNMStats(stats_path)
            self.from_data(exp_data_path, pore_data_path, stats)
            return

        print('Generating an artificial network (UNIMPLEMENTED)');
        # TODO: Get the graph generator and its arguments from the constructor's arguments
        self.generate_artificial_pnm(nx.watts_strogatz_graph)

    def generate_artificial_pnm(self, function):
        # TODO: Generate a graph in self.graph using function
        return

    def from_data(self, exp_data_path, pore_data_path, stats):
        print('Reading the experimental dataset at {}'.format(exp_data_path))
        dataset = xr.load_dataset(exp_data_path)
        label_matrix = dataset['label_matrix'].data
        labels = dataset['label'].data
        self.data = dataset

        print('Getting adjacency matrix for the experimental dataset')
        matrix = self.adjacency_matrix(label_matrix)
        # remove diagonal entries (self-loops)
        matrix[np.where(np.diag(np.ones(matrix.shape[0], dtype=np.bool)))] = False

        # remove irrelevant/noisy labels, pores that are just a few pixels large
        mask = np.ones(matrix.shape[0], np.bool)
        mask[labels] = False
        matrix[mask,:] = False
        matrix[:,mask] = False

        # fill networkx graph object
        coo_matrix = sp.sparse.coo_matrix(matrix)
        conn_list = zip(coo_matrix.row, coo_matrix.col)
        self.graph = nx.Graph()
        self.graph.add_edges_from(conn_list)

        # load pore properties
        print('Reading the pore network dataset at {}'.format(pore_data_path))
        pore = xr.load_dataset(pore_data_path)
        self.params['re'] = px*np.sqrt(pore['value_properties'].sel(property = 'median_area').data/np.pi)
        self.params['h0e'] = px*pore['value_properties'].sel(property = 'major_axis').data

        # define waiting times
        self.waiting_times = stats.delta_t_025

    def adjacency_matrix(self, label_im):
        def neighbour_search(label, struct=cube):
            mask = label_im==label
            mask = sp.ndimage.binary_dilation(input = mask, structure = struct(3))
            neighbours = np.unique(label_im[mask])[1:]
            return neighbours

        size = len(label_im)
        labels = np.unique(label_im[1:])
        matrix = np.zeros([size,size], dtype=np.bool)

        results = Parallel(n_jobs=job_count)(delayed(neighbour_search)(label) for label in labels)

        for (label, result) in zip(labels, results):
            matrix[label, result] = True

        # make sure that matrix is symmetric (as it should be)
        matrix = np.maximum(matrix, matrix.transpose())

        return matrix

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
    to_remove = deque()

    #  check if inlet pores are already filled
    filled_inlets = deque(inlets)
    for node in filled_inlets:
        if filled[node] == False:
            to_remove.append(node)

    for node in to_remove:
        filled_inlets.remove(node)

    # this part iteratively (should) calculate the effective inlet resistance
    # for every pore with the same distance (current_layer) to the network inlet
    current_layer = filled_inlets

    for node in current_layer:
        to_visit[node] = False

    while True in to_visit:
        next_layer = deque()
        for node in current_layer:
            for neighbour in net.neighbors(node):
                if neighbour not in next_layer:
                    next_layer.append(neighbour)

        for node in next_layer:
            R_nb = deque()
            nnbb = net.neighbors(node)
            for nb in nnbb:
                if nb in current_layer:
                    R_nb.append(R0[nb]+R_full[nb])

            R0[node] = effective_resistance(np.array(R_nb))
            to_visit[node] = False

        current_layer = next_layer

        to_remove.clear()
        for node in current_layer:
            if not filled[node]:
                to_remove.append(node)

        for node in to_remove:
            current_layer.remove(node)

    return R0

def simulation(pnm):
    # this part is necessary to match the network pore labels to the pore property arrays
    nodes = np.array(pnm.graph.nodes)
    n = len(nodes)  
    n_init = nodes.max()+1
    node_ids = nodes
    node_ids.sort()
    node_ids = np.array(node_ids)
    
    
    num_inlets = max(int(0.1*n),6)
    inlets = np.array(pnm.params['inlets'])
    if not np.any(inlets):
        inlets = np.random.choice(nodes, num_inlets)
        inlets = np.unique(inlets)
    temp_lets = []
    
    # double-check if inlet pores are actually in the network
    for inlet in inlets:
        if inlet in pnm.graph:
            temp_lets.append(inlet)
    inlets = temp_lets
    # print(inlets)


    # asign a random waiting time to every pore based on the experimental distribution
    if np.any(pnm.waiting_times):       
        ecdf = ECDF(pnm.waiting_times)
        f = interp1d(ecdf.y[1:], ecdf.x[1:], fill_value = 'extrapolate')
        prob = np.random.rand(n_init)
        t_w = f(prob)     
    #t_w = t_w*0  


    # create new pore property arrays where the pore label corresponds to the array index
    # this copuld be solved more elegantly with xarray, but the intention was that it works
    
    time = np.arange(pnm.params['t_init'], pnm.params['tmax'], pnm.params['dt'])
    act_time = np.zeros(n_init)
    filled = np.zeros(n_init, dtype = np.bool)
    h = np.zeros(n_init)+1E-6
    R0 = np.zeros(n_init)
    r = np.zeros(n_init)
    h0 = np.zeros(n_init)
    cc=0
    for node_id in node_ids:
        r[node_id] = pnm.params['re'][cc]
        h0[node_id] = pnm.params['h0e'][cc]
        cc=cc+1
    
    R0[inlets] = pnm.params['R_inlet']
    V=np.zeros(len(time))
    R_full = poiseuille_resistance(h0, r) +R0

    # this is the simulation:
    active = deque(inlets)
    newly_active = deque()
    finished = deque()

    # every time step solve explicitly
    tt=0
    for t in time:
        # first check, which pores are currently getting filled (active)
        if len(newly_active) > 0:
            for node in newly_active:
                act_time[node] = t + t_w[node]
                if not filled[node] and node not in active:
                    active.append(node)

            R0 = outlet_resistances(inlets, filled, R_full, pnm.graph)
            newly_active.clear()

        # calculate the new filling state (h) for every active pore
        for node in active:
            if t>act_time[node]:
                h_old = h[node]
                #h[node] = h[node] + dt*capillary_rise(t-act_time[node], r[node], R0[node])
                if node in inlets:
                    # patch to consider inlet resitance at inlet pores
                    h[node] = h_old + pnm.params['dt']*capillary_rise2(r[node], R0[node]+ pnm.params['R_inlet'], h_old)
                else:
                    # influence of inlet resistance on downstream pores considered by initialization of poiseuille resistances
                    h[node] = h_old + pnm.params['dt']*capillary_rise2(r[node], R0[node], h_old)

                # if pore is filled, look for neighbours that would now start to get filled
                if h[node] >= h0[node]:
                    h[node] = h0[node]
                    filled[node] = True
                    finished.append(node)
                    newly_active += pnm.graph.neighbors(node)

        for node in finished:
            active.remove(node)
        finished.clear()

        V[tt] = total_volume(h[node_ids], r[node_ids])
        tt=tt+1

    return [time, V]

def plot_results(pnm, results):
    for result in results:
        plt.loglog(result[0], result[1])   

    plt.title('experimental network')
    plt.xlabel('time [s]')
    plt.ylabel('volume [m3]')
    plt.xlim(0.1,pnm.params['tmax'])

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

    # compare to experimental data
    plt.figure()
    vxm3 = px**3
    test = np.array(results)
    std = test[:,1,:].std(axis=0)
    mean = test[:,1,:].mean(axis=0)
    time_line = test[0,0,:]

    # Configure the axes
    time_coarse = time_line[::1000]
    mean_coarse = mean[::1000]
    std_coarse = std[::1000]

    # Configure the plot
    plt.plot(time_coarse, mean_coarse)#)
    plt.fill_between(time_coarse, mean_coarse-std_coarse, mean_coarse+std_coarse, alpha=0.2)

    # Create data and plot it
    (pnm.data['volume'].sum(axis = 0)*vxm3).plot(color='k')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    ### Parse arguments
    parser = argparse.ArgumentParser(description = 'Simulation parameters')
    parser.add_argument('-G', '--generate-network', action = 'store_true', help = 'Generate an artificial pore network model and ignore -E, -P and -S')
    parser.add_argument('-c', '--iteration-count', type = int, default = 1, help = 'The amount of times to run the simulation (default to 1)')
    parser.add_argument('-j', '--job-count', type = int, default = job_count, help = 'The amount of jobs to use (default to {})'.format(job_count))
    parser.add_argument('-E', '--exp-data', default = None, help = 'Path to the experimental data')
    parser.add_argument('-P', '--pore-data', default = None, help = 'Path to the pore network data')
    parser.add_argument('-S', '--stats-data', default = None, help = 'Path to the network statistics')
    parser.add_argument('-Np', '--no-plot', action = 'store_true', help = 'Don\'t plot the results')

    args = parser.parse_args()
    if not args.generate_network and not all([args.exp_data, args.pore_data, args.stats_data]):
        raise ValueError('Either -G has to be used, or all of the data paths have to be defined.')
    if args.iteration_count < 0:
        raise ValueError('-c has to be greater or equal to 0.')
    if args.job_count <= 0:
        raise ValueError('-j has to be greater or equal to 1.')

    job_count = args.job_count # This is a global variable that remains constant from here

    ### Initialize the PNM
    results = []
    pnm = WickingPNM(args.generate_network, args.exp_data, args.pore_data, args.stats_data)
    pnm.params['R_inlet'] = 5E19 #Pas/m3
    pnm.params['inlets'] = [162, 171, 207]

    ### Get simulation results
    I = args.iteration_count
    if I == 0:
        # We just wanted to build the network
        sys.exit()
    if I == 1:
        print('Starting the simulation to run once.')
        results = [simulation(pnm)]
    else:
        njobs = min(I, job_count)
        print('Starting the simulation for {} times with {} jobs.'.format(I, njobs))
        with mp.Pool(njobs) as pool:
            results = pool.map(simulation, np.full(I, pnm))

    if not args.no_plot:
        plot_results(pnm, results)
