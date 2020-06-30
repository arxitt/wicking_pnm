import numpy as np
import matplotlib.pyplot as plt

from collections import deque

class Material:
    def __init__(self,
            eta = 1,
            gamma = 72.6,
            theta = 50,
            px = 2.75E-6
        ):
        # General physical constants (material-dependent)
        self.eta = eta # (mPa*s) dynamic viscosity of water
        self.gamma = gamma # (mN/m) surface tension of water
        self.theta = theta
        self.cos_theta = np.cos(np.radians(theta)) # Young's contact angle
        self.px = px # (m)

    def __str__(self):
        return "Material(eta = {}, gamma = {}, theta = {}, px = {})"\
            .format(self.eta, self.gamma, self.theta, self.px)

    ## function to calculate the resistance of a full pore
    def poiseuille_resistance(self, l, r):
        return 8*self.eta*l/np.pi/r**4

    ## function to calculate the filling velocity considering the inlet resistance and tube radius
    def capillary_rise(self, t, r, R0):
        gamma, cos, eta = self.gamma, self.cos_theta, self.eta
        return gamma*r*cos/2/eta/np.sqrt((R0*np.pi*r**4/8/eta)**2+gamma*r*cos*t/2/eta)

    ## use capillary_rise2 because the former did not consider that R0 can change over time, should be irrelevant because pore contribution becomes quickly irrelevant , but still...
    def capillary_rise2(self, r, R0, h):
        return 2*self.gamma*self.cos_theta/(R0*np.pi*r**3+8*self.eta*h/r)

    ## wrap up pore filling states to get total amount of water in the network
    def total_volume(self, h, r):
        return (h*np.pi*r**2).sum()

class Simulation:
    def __init__(self, pnm,
            material = Material(),
            sqrt_factor = 0,
            max_time = 1600,
            time_step = 1E-2,
            verbose = False
        ):
        self.pnm = pnm
        self.material = material
        self.sqrt_factor = sqrt_factor
        self.max_time = max_time
        self.time_step = time_step
        self.verbose = verbose

        # this part is necessary to match the network pore labels to the pore property arrays
        nodes = self.nodes = np.array(pnm.graph.nodes)
        self.n_init = nodes.max() + 1
        self.node_ids = np.array(nodes)
        self.node_ids.sort()

        # create new pore property arrays where the pore label corresponds to the array index
        # this copuld be solved more elegantly with xarray, but the intention was that it works
        n = self.n_init
        pnm.filled = np.zeros(n, dtype = np.bool)
        pnm.R0 = np.zeros(n)
        self.act_time = np.zeros(n)
        self.h = np.zeros(n)+1E-6
        self.r = np.zeros(n)
        self.h0 = np.zeros(n)

        i = 0
        for node_id in self.node_ids:
            self.r[node_id] = pnm.radi[i]
            self.h0[node_id] = pnm.heights[i]
            i += 1

        self.plot_sqrt = sqrt_factor > 0
        self.line_alpha = lambda _: 1
        if self.plot_sqrt:
            self.xsqrt = np.arange(1, self.max_time)
            self.ysqrt = sqrt_factor*np.sqrt(xsqrt)
            self.line_alpha = lambda results: \
                1 if len(results) < 10 else 0.5
            self.sqrt_col = 'chartreuse'

    # TODO: Move the initialization to __init__
    def run(self):
        pnm = self.pnm
        filled = pnm.filled
        R0 = pnm.R0
        act_time = self.act_time
        h = self.h
        r = self.r
        h0 = self.h0

        # Generate_waiting_times will build new waiting times with the same
        # technique, this is needed so that multiple runs (in multiple
        # processes) on the same PNM don't produce the same results.
        pnm.generate_waiting_times()

        inlets = pnm.inlets
        R0[inlets] = pnm.R_inlet
        pnm.R_full = self.material.poiseuille_resistance(h0, r) + R0

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
        tmax = self.max_time
        dt = np.float64(self.time_step)
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
                        h[node] += dt*self.material.capillary_rise2(r[node], R0[node] + R_inlet, h[node])
                    else:
                        # influence of inlet resistance on downstream pores considered by initialization of poiseuille resistances
                        h[node] += dt*self.material.capillary_rise2(r[node], R0[node], h[node])

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
            V[step] = self.material.total_volume(h[self.node_ids], r[self.node_ids])
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
        plt.xlim(0.1, self.max_time)
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
        plt.xlim(0.1, self.max_time)
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
        vxm3 = self.material.px**3
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

        if self.pnm.data is not None:
            (self.pnm.data['volume'].sum(axis = 0)*vxm3).plot(color='k')

        plt.title('Comparison between the absorbed volume and the experimental data')
        return plt

    def plot_all(self, results):
        self.plot_comparison(results)
        self.plot_simulation(results)
        self.plot_simulation_logarithmic(results)
        plt.show()
