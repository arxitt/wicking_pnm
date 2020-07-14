#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 08:19:34 2020

@author: firo
"""

import sys
import argparse

import numpy as np
import networkx as nx
from joblib import Parallel, delayed

from wickingpnm.model import PNM
from wickingpnm.simulation import Simulation, Material

# TODO: Add more
graph_functions = {
    'random_regular_graph': lambda n: nx.random_regular_graph(4, n)
}

if __name__ == '__main__':
    ### Parse arguments
    parser = argparse.ArgumentParser(description = 'Simulation parameters')
    parser.add_argument('-D', '--data-path', type = str, default = './data', help = 'Path to the data files (default to ./data)')
    parser.add_argument('-S', '--sample', type = str, default = 'T3_025_3_III', help = 'Sample name (default to T3_025_3_III)')
    parser.add_argument('-G', '--generate-network', type = str, default = '', help = 'Generate an artificial pore network model using the given function')
    parser.add_argument('-n', '--node-count', type = int, default = 100, help = 'The amount of nodes in the random graph (default to 100)')
    parser.add_argument('-m', '--material', type = str, default = '', help = 'Material parameters written as eta,gamma,theta,px')
    parser.add_argument('-f', '--sqrt-factor', type = float, default = 0, help = 'Show a square root with the given factor with the plots')
    parser.add_argument('-r', '--upstream-resistance', type = float, default = 2E17, help = 'Upstream resistance affecting the inlet pores (default to 2E17)')
    parser.add_argument('-c', '--iteration-count', type = int, default = 1, help = 'The amount of times to run the simulation (default to 1)')
    parser.add_argument('-j', '--job-count', type = int, default = 4, help = 'The amount of jobs to use (default to 4)')
    parser.add_argument('-s', '--time-step', type = float, default = 1E-3, help = 'The atomic time step to use throughout the simulation in seconds (default to 0.001)')
    parser.add_argument('-t', '--max-time', type = float, default = 1600, help = 'The amount of time to simulate in seconds (default to 1600)')
    parser.add_argument('-i', '--inlets', type = str, default = '', help = 'Labels for inlet pores (random by default, ignores -ci)')
    parser.add_argument('-In', '--inlet-nodes', type = str, default = '', help = 'Node names for inlet pores (only with 1D graphs, ignores -i and -ci)')
    parser.add_argument('-Ic', '--inlet-count', type = str, default = 5, help = 'The amount of inlet pores to generate (default to 5)')
    parser.add_argument('-Re', '--random-exp-data', action = 'store_true', help = 'Randomize the experimental data')
    parser.add_argument('-Rp', '--random-pore-props', action = 'store_true', help = 'Randomize the pore properties')
    parser.add_argument('-Rwt', '--random-waiting-times', action = 'store_true', help = 'Randomize the waiting times')
    parser.add_argument('-Np', '--no-plot', action = 'store_true', help = 'Don\'t plot the results')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Be verbose during the simulation')

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
    R_inlet = args.upstream_resistance # Pas/m3
    inlet_nodes = args.inlet_nodes.split(',') # Previously [162, 171, 207]
    inlets = args.inlets.split(',')
    if '' in inlets:
        inlets = []
    else:
        for inlet in inlets:
            inlet = int(inlet)

    pnm_params = {
        'data_path': args.data_path,
        'sample': args.sample,
        'inlets': inlets,
        'inlet_count': args.inlet_count,
        'R_inlet': R_inlet,
        'job_count': job_count,
        'rand_exp_data': args.random_exp_data,
        'rand_pore_props': args.random_pore_props,
        'rand_waiting_times': args.random_waiting_times,
        'verbose': verbose
    }

    if args.generate_network:
        if args.generate_network in graph_functions:
            n = args.node_count
            print('Generating an artificial network ({} - {} nodes)'.format(args.generate_network, n));
            pnm_params['graph'] = graph_functions[args.generate_network](n)
        else:
            raise ValueError('Invalid graph function passed to -G.')

    pnm = PNM(**pnm_params)

    if inlet_nodes:
        print('Using inlet names:', inlet_nodes)

        inlets = []
        for inlet in inlet_nodes:
            inlets.append(pnm.label_dict[int(inlet)])

        print('Got inlet labels:', inlets)

        pnm.inlets = inlets
        pnm.build_inlets()

    if verbose:
        print('\nre', pnm.radi, '\n')
        print('\nh0e', pnm.heights, '\n')
        print('\nwaiting times', pnm.waiting_times, '\n')
        print('\ninlets', pnm.inlets, '\n')

    if args.material:
        mp = []
        for param in args.material.split(','):
            mp.append(np.float64(param))

        material = Material(*mp)
    else:
        material = Material()

    print('Using material {}'.format(material));
    simulation = Simulation(pnm,
        material = material,
        sqrt_factor = args.sqrt_factor,
        max_time = args.max_time,
        time_step = args.time_step,
        verbose = verbose
    )

    ### Get simulation results
    I = args.iteration_count
    if I == 0:
        # We just wanted to build the network
        sys.exit()
    if I == 1:
        print('Starting the simulation to run once with a timestep of {}s.'.format(args.time_step))
        results = [simulation.run()]
    else:
        njobs = min(I, job_count)
        print('Starting the simulation with a timestep of {}s for {} times with {} jobs.'.format(args.time_step, I, njobs))
        results = Parallel(n_jobs=njobs)(delayed(simulation.run)() for i in range(I))

    if not args.no_plot:
        simulation.plot_all(results)

    # TODO: Make a way to store the data, ideally as hdf5/netcdf4
