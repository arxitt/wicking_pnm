import numpy as np
import matplotlib.pyplot as plt

from statsmodels.distributions.empirical_distribution import ECDF
from collections import deque

class Simulation:
    def __init__(self, pnm, sqrt_factor = 0, verbose = False):
        self.pnm = pnm
        self.sqrt_factor = sqrt_factor
        self.verbose = verbose

        self.plot_sqrt = sqrt_factor > 0
        self.line_alpha = lambda _: 1
        if self.plot_sqrt:
            self.xsqrt = np.arange(1, pnm.params['tmax'])
            self.ysqrt = sqrt_factor*np.sqrt(xsqrt)
            self.line_alpha = lambda results: \
                1 if len(results) < 10 else 0.5
            self.sqrt_col = 'chartreuse'

    # TODO: Move the initialization to __init__
    def run(self):
        pnm = self.pnm

        # this part is necessary to match the network pore labels to the pore property arrays
        nodes = np.array(pnm.graph.nodes)
        n = len(nodes)  
        n_init = nodes.max()+1
        node_ids = nodes
        node_ids.sort()
        node_ids = np.array(node_ids)

        # Generate_waiting_times will build new waiting times with the same
        # technique, this is needed so that multiple runs (in multiple
        # processes) on the same PNM don't produce the same results.
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
        R0[inlets] = pnm.R_inlet
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
        R_inlet = pnm.R_inlet
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
                    if self.verbose:
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
                if self.verbose:
                    print('\nnode finished\n', node)

                active.remove(node)
            finished.clear()

            time[step] = t
            # TODO: Stop when the filling slows down meaningfully
            V[step] = pnm.total_volume(h[node_ids], r[node_ids])
            step += 1
            t += dt

        return [time, V]
    
    ## TODO: Have the plot functions take an optional figure as argument

    def plot_simulation(self, results):
        plt.figure()
        for result in results:
            plt.plot(result[0], result[1], alpha = self.line_alpha(results))

        if self.plot_sqrt:
            plt.plot(self.xsqrt, self.ysqrt, dashes = (5, 5), color = self.sqrt_col, alpha = 1)
        plt.title('Absorbed volume for each run')
        plt.xlabel('time [s]')
        plt.ylabel('volume [m3]')
        plt.xlim(0.1, self.pnm.params['tmax'])
        return plt
        
    def plot_simulation_logarithmic(self, results):
        plt.figure()
        for result in results:
            plt.loglog(result[0], result[1], alpha = self.line_alpha(results))

        if self.plot_sqrt:
            plt.loglog(self.xsqrt, self.ysqrt, dashes = (5, 5), color = self.sqrt_col, alpha = 1)

        plt.title('Absorbed volume for each run (logarithmic)')
        plt.xlabel('time [s]')
        plt.ylabel('volume [m3]')
        plt.xlim(0.1, self.pnm.params['tmax'])
        return plt

    def plot_flux(self, results):
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
        return plt

    def plot_comparison(self, results):
        # compare to experimental data
        plt.figure()
        vxm3 = self.pnm.params['px']**3
        test = np.array(results)
        std = test[:,1,:].std(axis=0)
        mean = test[:,1,:].mean(axis=0)
        time_line = test[0,0,:]
        alpha = 0.2 if self.plot_sqrt else 1

        # Configure the axes
        time_coarse = time_line[::1000]
        mean_coarse = mean[::1000]
        std_coarse = std[::1000]

        # Configure the plot
        plt.plot(time_coarse, mean_coarse)
        plt.fill_between(time_coarse, mean_coarse-std_coarse, mean_coarse+std_coarse, alpha = alpha)

        if self.plot_sqrt:
            plt.plot(xsqrt, ysqrt, dashes = (5, 5), color = sqrt_col, alpha = 1)

        if self.pnm.data:
            (self.pnm.data['volume'].sum(axis = 0)*vxm3).plot(color='k')

        plt.title('Comparison between the absorbed volume and the experimental data')
        return plt

    def plot_all(self, results):
        self.plot_comparison(results)
        self.plot_simulation(results)
        self.plot_simulation_logarithmic(results)
        plt.show()
