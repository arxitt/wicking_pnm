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
from collections import deque, namedtuple
from skimage import measure
from skimage.morphology import cube

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

        self.randomize_waiting_times = False
        self.waiting_times_data = None

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

        # re and h0e are on a scale of 1E-6 to 1E-3, waiting times from 1 to 1000
        for i in range(size):
            re[i] /= 10**np.random.randint(3, 6)
            h0e[i] /= 10**np.random.randint(3, 6)

        self.build_inlets()
        self.generate_waiting_times()

    def extract_throat_list(self, label_matrix, labels, dt = None): 
        """
        inspired by Jeff Gostick's GETNET

        extracts a list of directed throats connecting pores i->j including a few throat parameters
        undirected network i-j needs to be calculated in a second step
        """

        def extend_bounding_box(s, shape, pad=3):
            a = []
            for i, dim in zip(s, shape):
                start = 0
                stop = dim

                if i.start - pad >= 0:
                    start = i.start - pad
                if i.stop + pad < dim:
                    stop = i.stop + pad

                a.append(slice(start, stop, None))

            return a

        parameters = ['pore_1', 'pore_2', 'index',
                      'x_extent', 'y_extent', 'z_extent',
                      'inscribed_radius', 'perimeter', 'area',
                      'shape_factor', 'X', 'Y', 'Z',
                      'major_axis', 'minor_axis', 'aspect_ratio']
        throat_parameters = namedtuple('ThroatParameters', parameters)
        def label_function(struct, pore_object, sub_dt, bounding_box, label):
            pore_im = pore_object == label
            im_w_throats = sp.ndimage.binary_dilation(input = pore_im, structure = struct(3))
            im_w_throats = im_w_throats*pore_object
            n_labels = np.unique(im_w_throats)[1:]
            t_params = deque()
            for n_label in n_labels: # check neighbor labels for connections, later take the mean of the
                                     # property of throat A->B and B->A as throat property A-B
                if n_label == label: continue
                # FIXME: what about multiple connections between two pores --> do a find objects and then make them as extra throats
                # what about inverse throats? can you match them again? maybe via position (com) --> seems to be fixed
                throat_im = (im_w_throats == n_label).astype(np.uint8)
                throat_im = measure.label(throat_im, connectivity=3)
                throat_props = measure.regionprops(throat_im)

                for prop in throat_props:
                    x_extent, y_extent, z_extent = prop.image.shape
                    conn = (label, n_label)
                    throat = np.where(prop.image)
                    rad_inscribed = sub_dt[throat].max()
                    area, perimeter = prop.area, np.count_nonzero(sub_dt[throat] < 2)
                    if perimeter > 0:
                        shape_factor = area/perimeter**2
                    else:
                        shape_factor = 0

                    com = sp.ndimage.measurements.center_of_mass(prop.image)
                    com += np.array([bounding_box[0].start, bounding_box[1].start, bounding_box[2].start]) 

                    inertia_tensor = measure._moments.inertia_tensor(prop.image)
                    in_tens_eig = np.linalg.eigvalsh(inertia_tensor)

                    major_axis = 4*np.sqrt(np.abs(in_tens_eig[-1]))
                    minor_axis = 4*np.sqrt(np.abs(in_tens_eig[0]))

                    aspect_ratio = major_axis/minor_axis

                    t_param = throat_parameters(
                        conn[0], conn[1], 0, x_extent, y_extent,
                        z_extent, rad_inscribed, perimeter, area,
                        shape_factor, com[0], com[1], com[2],
                        major_axis, minor_axis, aspect_ratio
                    )

                    t_params.append(t_param)

            return np.array(t_params)

        im = label_matrix

        struct = cube # FIXME: ball does not work as you would think (anisotropic expansion)
        if im.ndim == 2:
            struct = disk

        if dt == None:
            dt = sp.ndimage.distance_transform_edt(im>0)

        crude_pores = sp.ndimage.find_objects(im)
        # throw out None-entries (counterintuitive behavior of find_objects)
        pores = []
        for pore in crude_pores:
            if not pore == None:
                pores.append(pore)
        crude_pores = None

        shape = im.shape
        bounding_boxes = []
        for pore in pores:
            bounding_boxes.append(extend_bounding_box(pore, shape))

        t_params_raw = Parallel(n_jobs = job_count, backend = 'threading')(
            delayed(label_function)\
                (struct, im[bounding_box], dt[bounding_box], bounding_box, label) \
                for (bounding_box, label) in zip(bounding_boxes, labels)
        )

        # clear out empty objects
        t_params = []
        for param in t_params_raw:
            if len(param) > 0:
                t_params.append(param)

        return np.concatenate(t_params, axis = 0)

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
        throats = self.extract_throat_list(label_matrix, labels)

        self.graph = nx.Graph()
        self.graph.add_edges_from(np.uint16(throats[:,:2]))
        self.params['re'] = np.sqrt(throats[:,8]/np.pi)*self.params['px']
        self.params['h0e'] = throats[:,-3]*self.params['px']

        # define waiting times
        self.waiting_times_data = stats.delta_t_all
        self.generate_waiting_times()
        self.build_inlets()

    def generate_waiting_times(self):
        size = np.array(np.unique(self.graph.nodes)).max() + 1
        data = self.waiting_times_data

        if data is not None and len(data) > 0:
            # assign a random waiting time to every pore based on the experimental distribution
            ecdf = ECDF(data)
            func = interp1d(ecdf.y[1:], ecdf.x[1:], fill_value = 'extrapolate')
            self.waiting_times = func(np.random.rand(size))
        else:
            wt = self.waiting_times = np.random.rand(size)
            for i in range(size):
                wt[i] *= 10**np.random.randint(0, 3)

    def build_inlets(self, amount = 5):
        inlets = np.array(pnm.inlets, dtype = np.int)
        if not np.any(inlets):
            self.generate_inlets(amount)
        else:
            # double-check if inlet pores are actually in the network
            temp_inlets = deque()
            print('Taking inlets from command-line arguments.')
            for inlet in inlets:
                if inlet in pnm.graph:
                    temp_inlets.append(inlet)
            pnm.inlets = np.array(temp_inlets)

    # TODO: Change this to start with one random inlet and some amount of distant neighbours
    def generate_inlets(self, amount):
        print('Generating {} inlets'.format(amount))
        pnm.inlets = np.unique(np.random.choice(np.unique(self.graph.nodes), amount))

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

    # this function recursivelself.waiting_times ould calculate the effective inlet resistance
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

    # Generate_waiting_times will build new waiting times with the ECDF distribution
    if pnm.randomize_waiting_times:
        pnm.generate_waiting_times()

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
    t_w = pnm.waiting_times
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

def plot(pnm, results, sqrt_factor = 0):
    plot_sqrt = sqrt_factor > 0
    if plot_sqrt:
        xsqrt = np.arange(1, pnm.params['tmax'])
        ysqrt = sqrt_factor*np.sqrt(xsqrt)
        line_alpha = 1 if len(results) < 10 else 0.5
        sqrt_col = 'chartreuse'

    def plot_simulation_logarithmic():
        alpha = 0.4 if plot_sqrt else 1
        plt.figure()
        for result in results:
            plt.loglog(result[0], result[1], alpha = alpha)

        if plot_sqrt:
            plt.loglog(xsqrt, ysqrt, dashes = (5, 5), color = sqrt_col, alpha = 1)

        plt.title('Absorbed volume for each run (logarithmic)')
        plt.xlabel('time [s]')
        plt.ylabel('volume [m3]')
        plt.xlim(0.1,pnm.params['tmax'])

    def plot_simulation():
        alpha = 0.4 if plot_sqrt else 1
        plt.figure()
        for result in results:
            plt.plot(result[0], result[1], alpha = alpha)

        if plot_sqrt:
            plt.plot(xsqrt, ysqrt, dashes = (5, 5), color = sqrt_col, alpha = 1)
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
    
        plt.title('Flux through the pore network')
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
        alpha = 0.2 if plot_sqrt else 1

        # Configure the axes
        time_coarse = time_line[::1000]
        mean_coarse = mean[::1000]
        std_coarse = std[::1000]

        # Configure the plot
        plt.plot(time_coarse, mean_coarse)
        plt.fill_between(time_coarse, mean_coarse-std_coarse, mean_coarse+std_coarse, alpha = alpha)

        if plot_sqrt:
            plt.plot(xsqrt, ysqrt, dashes = (5, 5), color = sqrt_col, alpha = 1)

        if pnm.data:
            (pnm.data['volume'].sum(axis = 0)*vxm3).plot(color='k')

        plt.title('Comparison between the absorbed volume and the experimental data')

    plot_comparison()
    plot_simulation()
    plot_simulation_logarithmic()
    plt.show()

if __name__ == '__main__':
    ### Parse arguments
    parser = argparse.ArgumentParser(description = 'Simulation parameters')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Be verbose during the simulation')
    parser.add_argument('-G', '--generate-network', action = 'store_true', help = 'Generate an artificial pore network model and ignore -E, -P and -S')
    parser.add_argument('-f', '--sqrt-factor', type = float, default = 0, help = 'Show a square with the given factor with the plots')
    parser.add_argument('-c', '--iteration-count', type = int, default = 1, help = 'The amount of times to run the simulation (default to 1)')
    parser.add_argument('-s', '--time-step', type = float, default = 1E-3, help = 'The atomic time step to use throughout the simulation in seconds (default to 0.001)')
    parser.add_argument('-t', '--max-time', type = float, default = 1600, help = 'The amount of time to simulate in seconds (default to 1600)')
    parser.add_argument('-n', '--node-count', type = int, default = 100, help = 'The amount of nodes in the random graph (default to 100)')
    parser.add_argument('-i', '--inlets', type = str, default = '', help = 'Labels for inlet pores (random by default)')
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
    pnm.inlets = args.inlets.split(',') # Previously [162, 171, 207]
    if '' in pnm.inlets:
        pnm.inlets = []
    else:
        for inlet in pnm.inlets:
            inlet = int(inlet)

    if args.generate_network:
        print('Generating an artificial network');
        pnm.generate(nx.random_regular_graph, 4, args.node_count)
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
        pnm.randomize_waiting_times = True
        results = Parallel(n_jobs=njobs)(delayed(simulation)(pnm) for i in range(I))

    if not args.no_plot:
        plot(pnm, results, args.sqrt_factor)
