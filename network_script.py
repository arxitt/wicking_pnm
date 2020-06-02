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
from collections import deque, namedtuple

job_count = 4 # Default job count, 4 should be fine on most systems
verbose = False

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
    def __init__(self):
        self.data = None
        self.graph = None
        self.waiting_times = np.array([])
        self.V = None # Water volume in the network
        self.filled = None # filled nodes
        self.inlets = None # inlet pores
        self.R0 = None # pore resistances
        self.R_full = None # resistances when full
        self.R_inlet = 0 # inlet resistance

        self.params = {
            # General physical constants (material-dependent)
            'eta': 1, # (mPa*s) dynamic viscosity of water
            'gamma': 72.6, # (mN/m) surface tension of water
            'cos_theta': np.cos(np.radians(50)), # Young's contact angle
            'px': 2.75E-6, # (m)
            

            # Simulation boundaries
            'tmax': 1600, # (seconds)
            'dt': 1E-3, # (seconds), I think it's safe to increase it a bit, maybe x10-100

            # Experimental radius and height
            're': 0,
            'h0e': 0,
        }

    def generate(self, function, *args):
        graph = self.graph = function(*args)
        size = len(graph.nodes)
        print('Generated graph of size {}, filling with random data'.format(size))

        re = self.params['re'] = np.random.rand(size)
        h0e = self.params['h0e'] = np.random.rand(size)
        wt = self.waiting_times = np.random.rand(size)

        # re and h0e are on a scale of 1E-6 to 1E-3, waiting times from 1 to 1000
        for i in range(size):
            re[i] /= 10**np.random.randint(3, 6)
            h0e[i] /= 10**np.random.randint(3, 6)
            wt[i] *= 10**np.random.randint(0, 3)

        self.build_inlets()

    def from_data(self, exp_data_path, pore_data_path, stats_path):
        stats = WickingPNMStats(stats_path)

        print('Reading the experimental dataset at {}'.format(exp_data_path))
        dataset = xr.load_dataset(exp_data_path)
        label_matrix = dataset['label_matrix'].data
        labels = dataset['label'].data
        self.data = dataset

        if verbose:
            print('labels', labels)
            print('label matrix', label_matrix, label_matrix.shape)

        print('Generating the adjacency matrix for the experimental dataset')
        size = len(label_matrix)
        matrix = np.zeros((size, size), dtype = np.bool)
        slices = sp.ndimage.find_objects(label_matrix)
        # Get the parametric surface in euclidian space for each pore and analyze it
        def analyze_slice(slice):
            region = label_matrix[slice]
            nodes = np.unique(region)
            (x, y, z) = region.shape
            # Y, X = np.meshgrid(np.arange(y), np.arange(x)) # for plotting

            surfaces = {}
            for node in nodes:
                if node == 0 or node in surfaces:
                    continue

                # Parametric dataset for the pore's surface
                surface = namedtuple('Surface', ['min', 'max', 'limits'])
                surface.min = np.zeros((x, y))
                surface.max = np.zeros((x, y))
                surface.limits = {
                    'xmin': -1,
                    'xmax': -1,
                    'ymin': -1,
                    'ymax': -1,
                    'zmin': -1,
                    'zmax': -1,
                }

                surfaces[node] = surface

            if verbose:
                print('Nodes in region:', surfaces.keys())

            for i in range(x):
                for j in range(y):
                    labels = region[i, j]
                    unique_labels = np.unique(labels)

                    if len(unique_labels) == 1 and unique_labels[0] == 0:
                        continue

                    # Get maximum and minimum heights for region[i][j]
                    for k in range(z):
                        if labels[k] != 0:
                            surface = surfaces[labels[k]]
                            if surface.min[i, j] == -1:
                                surface.min[i, j] = k
                            if k > surface.max[i, j]: 
                                surface.max[i, j] = k

                    for surface in surfaces.values():
                        if surface.max[i, j] != -1:
                            limits = surface.limits
                            min, max = surface.min[i, j], surface.max[i, j]
                            if limits['xmin'] == -1:
                                limits['xmin'] = i
                            if limits['ymin'] == -1:
                                limits['ymin'] = j
                            if i > limits['xmax']:
                                limits['xmax'] = i
                            if j > limits['ymax']:
                                limits['ymax'] = j
                            if limits['zmin'] == -1 or min < limits['zmin']:
                                limits['zmin'] = surface.min[i, j]
                            if limits['zmax'] == -1 or max > limits['zmax']:
                                limits['zmax'] = surface.max[i, j]

            return surfaces

        surfaces_list = Parallel(n_jobs = job_count) \
            (delayed(analyze_slice)(slice) for slice in slices)

        for surfaces in surfaces_list:
            visited = {}
            for (node, surface) in surfaces.items():
                if verbose:
                    print(node, surface.limits)

                for (other_node, other_surface) in surfaces.items():
                    if other_node != node and other_node not in visited:
                        # TODO: Only do this operation where the surfaces overlap, if they do
                        delta1 = surface.min - other_surface.max # we're above
                        delta2 = surface.max - other_surface.min # we're below

                        if verbose:
                            print('\t', other_node, other_surface.limits)

                        if 1 in delta1 or 1 in delta2:
                            matrix[node, other_node] = True
                            matrix[other_node, node] = True

                            if verbose:
                                print('\t' + str(node) + ' connects to ' + str(other_node) + '.')

                visited[node] = True

        if verbose:
            print('adjacency matrix', matrix)

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
        px = pore.attrs['voxel'].data
        self.params['re'] = px*np.sqrt(pore['value_properties'].sel(property = 'median_area').data/np.pi)
        self.params['h0e'] = px*pore['value_properties'].sel(property = 'major_axis').data

        # define waiting times
        self.waiting_times = stats.delta_t_025
        self.build_inlets()

    def build_inlets(self):
        inlets = np.array(pnm.inlets)
        if not np.any(inlets):
            self.generate_inlets()
        else:
            # double-check if inlet pores are actually in the network
            temp_inlets = deque()
            print('Taking inlets from command-line arguments.')
            for inlet in inlets:
                if inlet in pnm.graph:
                    temp_inlets.append(inlet)
            pnm.inlets = np.array(temp_inlets)

    def generate_inlets(self):
        ninlets = max(int(0.1*len(self.graph.nodes)),6)
        print('Generating {} inlets'.format(ninlets))
        pnm.inlets = np.unique(np.random.choice(self.graph.nodes, ninlets))

    """
    find your path through the filled network to calculate the inlet
    resistance imposed on the pores at the waterfront
    quick and dirty, this part makes the code slow and might even be wrong
    we have to check
    """
    def outlet_resistances(self):
        # initialize pore resistances
        self.R0 = np.zeros(len(self.filled))

        # only filled pores contribute to the network permeability
        filled_inlets = deque()
        for inlet in self.inlets:
            if self.filled[inlet]:
                filled_inlets.append(inlet)

        if verbose:
            print('\nfilled inlets', filled_inlets)

        return self.outlet_resistances_r(filled_inlets)

    # this function recursively should calculate the effective inlet resistance
    # for every pore with the same distance (layer) to the network inlet
    def outlet_resistances_r(self, layer, visited = {}):
        if len(layer) == 0:
            return self.R0

        if verbose:
            print('current layer', layer)

        next_layer = deque()
        for node in layer:
            neighbours = self.graph.neighbors(node)
            inv_R_eff = np.float64(0)

            if verbose:
                print('visiting node', node)

            for neighbour in neighbours:
                if neighbour in layer:
                    inv_R_eff += 1/np.float64(self.R0[neighbour] + pnm.R_full[neighbour])

            self.R0[node] += 1/inv_R_eff

            if self.filled[node] and node not in visited:
                next_layer.append(node)

            visited[node] = True

        if verbose:
            print('next layer', next_layer)

        return self.outlet_resistances_r(next_layer, visited)

    ## function to calculate the resistance of a full pore
    def poiseuille_resistance(self, l, r):
        p = self.params
        return 8*p['eta']*l/np.pi/r**4

    ## function to calculate the filling velocity considering the inlet resistance and tube radius
    def capillary_rise(self, t, r, R0):
        p = self.params
        gamma, cos, eta = p['gamma'], p['cos_theta'], p['eta']
        return gamma*r*cos/2/eta/np.sqrt((R0*np.pi*r**4/8/eta)**2+gamma*r*cos*t/2/eta)

    ## use capillary_rise2 because the former did not consider that R0 can change over time, should be irrelevant because pore contribution becomes quickly irrelevant , but still...
    def capillary_rise2(self, r, R0, h):
        p = self.params
        return 2*p['gamma']*p['cos_theta']/(R0*np.pi*r**3+8*p['eta']*h/r)

    ## wrap up pore filling states to get total amount of water in the network
    def total_volume(self, h, r):
        return (h*np.pi*r**2).sum()

def simulation(pnm):
    # this part is necessary to match the network pore labels to the pore property arrays
    nodes = np.array(pnm.graph.nodes)
    n = len(nodes)  
    n_init = nodes.max()+1
    node_ids = nodes
    node_ids.sort()
    node_ids = np.array(node_ids)

    # asign a random waiting time to every pore based on the experimental distribution
    if np.any(pnm.waiting_times):       
        ecdf = ECDF(pnm.waiting_times)
        f = interp1d(ecdf.y[1:], ecdf.x[1:], fill_value = 'extrapolate')
        prob = np.random.rand(n_init)
        t_w = f(prob)
    #t_w = t_w*0

    # create new pore property arrays where the pore label corresponds to the array index
    # this copuld be solved more elegantly with xarray, but the intention was that it works

    filled = pnm.filled = np.zeros(n_init, dtype = np.bool)
    R0 = pnm.R0 = np.zeros(n_init)
    act_time = np.zeros(n_init)
    h = np.zeros(n_init)+1E-6
    r = np.zeros(n_init)
    h0 = np.zeros(n_init)
    cc=0
    for node_id in node_ids:
        r[node_id] = pnm.params['re'][cc]
        h0[node_id] = pnm.params['h0e'][cc]
        cc=cc+1

    inlets = pnm.inlets
    R0[inlets] = pnm.params['R_inlet']
    pnm.R_full = pnm.poiseuille_resistance(h0, r) + R0

    # this is the simulation:
    active = deque(inlets)
    newly_active = deque()
    finished = deque()

    # Create a dictionary for quicker lookup of whether the node is an inlet
    inlets_dict = {}
    for node in inlets:
        inlets_dict[node] = True

    # every time step solve explicitly
    R_inlet = pnm.params['R_inlet']
    tmax = pnm.params['tmax']
    dt = np.float64(pnm.params['dt'])
    t = dt

    step = 0
    time = np.zeros(np.int(np.ceil(tmax/dt)))
    V = pnm.V = np.zeros(len(time))
    while t <= tmax:
        # first check, which pores are currently getting filled (active)
        if len(newly_active) != 0:
            R0 = pnm.outlet_resistances()

        for node in newly_active:
            act_time[node] = t + t_w[node]
            if not filled[node] and node not in active:
                if verbose:
                    print('\nnew active node\n', node)

                active.append(node)
        newly_active.clear()

        # calculate the new filling state (h) for every active pore
        for node in active:
            if t > act_time[node]:
                if node in inlets_dict:
                    # patch to consider inlet resitance at inlet pores
                    h[node] += dt*pnm.capillary_rise2(r[node], R0[node] + R_inlet, h[node])
                else:
                    # influence of inlet resistance on downstream pores considered by initialization of poiseuille resistances
                    h[node] += dt*pnm.capillary_rise2(r[node], R0[node], h[node])

                # if pore is filled, look for neighbours that would now start to get filled
                if h[node] >= h0[node]:
                    h[node] = h0[node]
                    filled[node] = True
                    finished.append(node)
                    newly_active += pnm.graph.neighbors(node)

        for node in finished:
            if verbose:
                print('\nnode finished\n', node)

            active.remove(node)
        finished.clear()

        time[step] = t
        # TODO: Stop when the filling slows down meaningfully
        V[step] = pnm.total_volume(h[node_ids], r[node_ids])
        step += 1
        t += dt

    return [time, V]

def plot(pnm, results):
    def plot_simulation_logarithmic():
        for result in results:
            plt.loglog(result[0], result[1])   

        plt.title('simulation results')
        plt.xlabel('time [s]')
        plt.ylabel('volume [m3]')
        plt.xlim(0.1,pnm.params['tmax'])

    def plot_simulation():
        plt.figure()
        for result in results:
            plt.plot(result[0], result[1])   
        plt.title('Absorbed volume for each run')
        plt.xlabel('time [s]')
        plt.ylabel('volume [m3]')

    def plot_flux():
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

    def plot_comparison():
        # compare to experimental data
        plt.figure()
        vxm3 = pnm.params['px']**3
        test = np.array(results)
        std = test[:,1,:].std(axis=0)
        mean = test[:,1,:].mean(axis=0)
        time_line = test[0,0,:]

        # Configure the axes
        time_coarse = time_line[::1000]
        mean_coarse = mean[::1000]
        std_coarse = std[::1000]

        xhalf = np.array(range(1, 1000))
        yhalf = 1E-7*np.sqrt(xhalf)

        # Configure the plot
        plt.title('Comparison of absorbed volume in multiple runs')
        plt.loglog(time_coarse, mean_coarse)#)
        plt.loglog(xhalf, yhalf)
        plt.fill_between(time_coarse, mean_coarse-std_coarse, mean_coarse+std_coarse, alpha=0.2)

        if pnm.data:
            (pnm.data['volume'].sum(axis = 0)*vxm3).plot(color='k')

    plot_simulation()
    plot_comparison()
    plt.show()

if __name__ == '__main__':
    ### Parse arguments
    parser = argparse.ArgumentParser(description = 'Simulation parameters')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Be verbose during the simulation')
    parser.add_argument('-G', '--generate-network', action = 'store_true', help = 'Generate an artificial pore network model and ignore -E, -P and -S')
    parser.add_argument('-c', '--iteration-count', type = int, default = 1, help = 'The amount of times to run the simulation (default to 1)')
    parser.add_argument('-s', '--time-step', type = float, default = 1E-3, help = 'The atomic time step to use throughout the simulation in seconds (default to 0.001)')
    parser.add_argument('-t', '--max-time', type = float, default = 1600, help = 'The amount of time to simulate in seconds (default to 1600)')
    parser.add_argument('-n', '--node-count', type = int, default = 100, help = 'The amount of nodes in the random graph (default to 100)')
    parser.add_argument('-i', '--inlets', type = list, default = [], help = 'Labels for inlet pores (random by default)')
    parser.add_argument('-j', '--job-count', type = int, default = job_count, help = 'The amount of jobs to use (default to {})'.format(job_count))
    parser.add_argument('-E', '--exp-data', default = None, help = 'Path to the experimental data')
    parser.add_argument('-P', '--pore-data', default = None, help = 'Path to the pore network data')
    parser.add_argument('-S', '--stats-data', default = None, help = 'Path to the network statistics')
    parser.add_argument('-Np', '--no-plot', action = 'store_true', help = 'Don\'t plot the results')

    args = parser.parse_args()
    if args.iteration_count < 0:
        raise ValueError('-c has to be greater or equal to 0.')
    if args.job_count <= 0:
        raise ValueError('-j has to be greater or equal to 1.')

    # These are global variables that remain constant from here
    job_count = args.job_count
    verbose = args.verbose

    ### Initialize the PNM
    results = []
    pnm = WickingPNM()
    pnm.params['dt'] = args.time_step
    pnm.params['tmax'] = args.max_time
    pnm.params['R_inlet'] = np.int(2E17) #Pas/m3
    pnm.inlets = args.inlets # Previously [162, 171, 207]

    if args.generate_network:
        print('Generating an artificial network');
        pnm.generate(nx.newman_watts_strogatz_graph, args.node_count, 2, 0.2)
    elif all([args.exp_data, args.pore_data, args.stats_data]):
        print('Reading the network from data')
        pnm.from_data(args.exp_data, args.pore_data, args.stats_data)
    else:
        raise ValueError('Either -G has to be used, or all of the data paths have to be defined.')

    if verbose:
        print('\nre', pnm.params['re'], '\n')
        print('\nh0e', pnm.params['h0e'], '\n')
        print('\nwaiting times', pnm.waiting_times, '\n')
        print('\ninlets', pnm.inlets, '\n')

    ### Get simulation results
    I = args.iteration_count
    if I == 0:
        # We just wanted to build the network
        sys.exit()
    if I == 1:
        print('Starting the simulation to run once with a timestep of {}s.'.format(args.time_step))
        results = [simulation(pnm)]
    else:
        njobs = min(I, job_count)
        print('Starting the simulation with a timestep of {}s for {} times with {} jobs.'.format(args.time_step, I, njobs))
        results = Parallel(n_jobs=njobs)(delayed(simulation)(pnm) for i in range(I))

    if not args.no_plot:
        plot(pnm, results)
