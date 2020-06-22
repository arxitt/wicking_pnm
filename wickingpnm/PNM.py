import random
import xarray as xr
import numpy as np
import scipy as sp
import networkx as nx

from collections import deque, namedtuple
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from skimage.morphology import cube

def label_function(struct, pore_object, label):
    mask = pore_object == label
    connections = deque()

    if self.verbose:
        print('Searching around {}'.format(label))

    mask = sp.ndimage.binary_dilation(input = mask, structure = struct(3))
    neighbors = np.unique(pore_object[mask])[1:]

    for nb in neighbors:
        if nb != label:
            conn = (label, nb)

            if self.verbose:
                print('\t{} connects to {}'.format(conn[1], conn[0]))

            connections.append(conn)

    return connections

class PNMStats:
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

# TODO: Make an ExpPNM class and a ArtPNM class
class PNM:
    def __init__(self, job_count = 4, verbose = False):
        self.job_count = job_count
        self.verbose = verbose

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
        # we will need a method to pass experimental information on the pore sizes in the future analogous to 'generate_waiting_times'
        for i in range(size):
            re[i] /= 10**np.random.randint(5, 6)
            h0e[i] /= 10**np.random.randint(4, 5)

        self.build_inlets()
        self.generate_waiting_times()

    def extract_throat_list(self, label_matrix, labels): 
        """
        inspired by Jeff Gostick's GETNET

        extracts a list of directed throats connecting pores i->j including a few throat parameters
        undirected network i-j needs to be calculated in a second step
        """

        def extend_bounding_box(s, shape, pad=3):
            a = deque()
            for i, dim in zip(s, shape):
                start = 0
                stop = dim

                if i.start - pad >= 0:
                    start = i.start - pad
                if i.stop + pad < dim:
                    stop = i.stop + pad

                a.append(slice(start, stop, None))

            return tuple(a)

        im = label_matrix

        struct = cube # FIXME: ball does not work as you would think (anisotropic expansion)
        if im.ndim == 2:
            struct = disk

        crude_pores = sp.ndimage.find_objects(im)

        # throw out None-entries (counterintuitive behavior of find_objects)
        pores = deque()
        bounding_boxes = deque()
        for pore in crude_pores:
            bb = extend_bounding_box(pore, im.shape)
            if pore is not None and len(np.unique(im[bb])) > 2:
                pores.append(pore)
                bounding_boxes.append(bb)

        connections_raw = Parallel(n_jobs = self.job_count)(
            delayed(label_function)\
                (struct, im[bounding_box], label) \
                for (bounding_box, label) in zip(bounding_boxes, labels)
        )

        # clear out empty objects
        connections = deque()
        for connection in connections_raw:
            if len(connection) > 0:
                connections.append(connection)

        return np.concatenate(connections, axis = 0)

    def from_data(self, exp_data_path, pore_data_path, stats_path):
        stats = WickingPNMStats(stats_path)

        print('Reading the experimental dataset at {}'.format(exp_data_path))
        dataset = xr.load_dataset(exp_data_path)
        label_matrix = dataset['label_matrix'].data
        labels = dataset['label'].data
        self.data = dataset

        if self.verbose:
            print('labels', labels)
            print('label matrix', label_matrix, label_matrix.shape)

        print('Generating the pore network graph from the experimental dataset')
        throats = self.extract_throat_list(label_matrix, labels)
        self.graph = nx.Graph()
        self.graph.add_edges_from(np.uint16(throats[:,:2]))

        ## From the throats
        # self.params['re'] = np.sqrt(throats[:,8]/np.pi)*self.params['px']
        # self.params['h0e'] = throats[:,-3]*self.params['px']
        ## From the pore dataset
        pore = xr.load_dataset(pore_data_path)
        px = pore.attrs['voxel'].data
        self.params['re'] = px*np.sqrt(pore['value_properties'].sel(property = 'median_area').data/np.pi)
        self.params['h0e'] = px*pore['value_properties'].sel(property = 'major_axis').data

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
                wt[i] *= 10**np.random.randint(-1, 3)

    def build_inlets(self, amount = 5):
        inlets = np.array(self.inlets, dtype = np.int)
        if not np.any(inlets):
            self.generate_inlets(amount)
        else:
            # double-check if inlet pores are actually in the network
            temp_inlets = deque()
            print('Taking inlets from command-line arguments.')
            for inlet in inlets:
                if inlet in self.graph:
                    temp_inlets.append(inlet)
            self.inlets = np.array(temp_inlets)

    # TODO: Change this to start with one random inlet and some amount of distant neighbours
    def generate_inlets(self, amount):
        print('Generating {} inlets'.format(amount))
        self.inlets = random.sample(self.graph.nodes, amount)

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

        if self.verbose:
            print('\nfilled inlets', filled_inlets)

        return self.outlet_resistances_r(filled_inlets)

    # this function recursivelself.waiting_times ould calculate the effective inlet resistance
    # for every pore with the same distance (layer) to the network inlet
    def outlet_resistances_r(self, layer, visited = {}):
        if len(layer) == 0:
            return self.R0

        if self.verbose:
            print('current layer', layer)

        next_layer = deque()
        for node in layer:
            neighbours = self.graph.neighbors(node)
            inv_R_eff = np.float64(0)

            if self.verbose:
                print('visiting node', node)

            for neighbour in neighbours:
                if neighbour in layer:
                    inv_R_eff += 1/np.float64(self.R0[neighbour] + self.R_full[neighbour]) #<- you sure about this? you add a resistance to an inverse resistance

            self.R0[node] += 1/inv_R_eff

            if self.filled[node] and node not in visited:
                next_layer.append(node)

            visited[node] = True

        if self.verbose:
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
