#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 08:19:34 2020

@author: firo
"""

import sys
import argparse

import networkx as nx
from joblib import Parallel, delayed

from wickingpnm.model import PNM
from wickingpnm.simulation import Simulation

if __name__ == '__main__':
    ### Parse arguments
    parser = argparse.ArgumentParser(description = 'Simulation parameters')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Be verbose during the simulation')
    parser.add_argument('-G', '--generate-network', action = 'store_true', help = 'Generate an artificial pore network model and ignore -E, -P and -S')
    parser.add_argument('-f', '--sqrt-factor', type = float, default = 0, help = 'Show a square root with the given factor with the plots')
    parser.add_argument('-n', '--node-count', type = int, default = 100, help = 'The amount of nodes in the random graph (default to 100)')
    parser.add_argument('-c', '--iteration-count', type = int, default = 1, help = 'The amount of times to run the simulation (default to 1)')
    parser.add_argument('-s', '--time-step', type = float, default = 1E-3, help = 'The atomic time step to use throughout the simulation in seconds (default to 0.001)')
    parser.add_argument('-t', '--max-time', type = float, default = 1600, help = 'The amount of time to simulate in seconds (default to 1600)')
    parser.add_argument('-R', '--upstream-resistance', type = int, default = 2E17, help = 'Upstream resistance affecting the inlet pores (default to 2E17)')
    parser.add_argument('-i', '--inlets', type = str, default = '', help = 'Labels for inlet pores (random by default)')
    parser.add_argument('-j', '--job-count', type = int, default = 4, help = 'The amount of jobs to use (default to 4)')
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
    R_inlet = args.upstream_resistance # Pas/m3
    inlets = args.inlets.split(',') # Previously [162, 171, 207]
    if '' in inlets:
        inlets = []
    else:
        for inlet in inlets:
            inlet = int(inlet)

    pnm_params = {
        'exp_data_path': args.exp_data,
        'pore_data_path': args.pore_data,
        'inlets': inlets,
        'R_inlet': R_inlet,
        'job_count': job_count,
        'verbose': verbose
    }

    if args.stats_data is None:
        print('No network statistics were given, using random waiting times.')

    if args.generate_network:
        print('Generating an artificial network');
        n = args.node_count
        pnm_params['graph'] = nx.random_regular_graph(4, n)
    elif pnm_params['exp_data_path'] is None:
        raise ValueError('Please use either -G or -E to choose a graph model')

    pnm = PNM(args.stats_data, **pnm_params)

    if verbose:
        print('\nre', pnm.radi, '\n')
        print('\nh0e', pnm.heights, '\n')
        print('\nwaiting times', pnm.waiting_times, '\n')
        print('\ninlets', pnm.inlets, '\n')

    simulation = Simulation(pnm,
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
