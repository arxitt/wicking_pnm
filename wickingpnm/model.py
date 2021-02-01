import sys

homeCodePath=r"H:\10_Python\005_Scripts_from_others\Laurent\wicking_pnm"
if homeCodePath not in sys.path:
	sys.path.append(homeCodePath)


import random
import xarray as xr
import numpy as np
import scipy as sp
import networkx as nx
import time

from collections import deque
from skimage.morphology import cube
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from joblib import Parallel, delayed
from os import path

import wickingpnm.waitingtimes as waitingtimes

time_limit = {'T3_100_10_III': 344,
              'T3_300_5': 229,
              'T3_100_7': 206,
              'T3_100_10': 232}

def label_function(struct, pore_object, label, labels, verbose = False):
    mask = pore_object == label
    connections = deque()

    if verbose:
        print('Searching around {}'.format(label))

    mask = sp.ndimage.binary_dilation(input = mask, structure = struct(3))
    neighbors = np.unique(pore_object[mask])[1:]

    for nb in neighbors:
        if nb != label:
            if nb in labels:
                conn = (label, nb)
    
                if verbose:
                    print('\t{} connects to {}'.format(conn[1], conn[0]))
    
                connections.append(conn)

    return connections

class PNM:
    def __init__(self,
        graph = None,
        data_path = None,
        sample = None,
        inlets = [],
        inlet_count = 5,
        R_inlet = 1E17,
        job_count = 4,
        rand_exp_data = False,
        rand_pore_props = False,
        rand_waiting_times = False,
        verbose = False,
        seed = int(time.time()),
        randomize_pore_data = False
    ):
        self.job_count = job_count
        self.verbose = verbose

        self.data_path = data_path # Head directory for the data
        self.sample = sample # Sample name
        
        self.seed = seed
        self.randomize_pore_data = randomize_pore_data

        dyn_data_dir, pore_props_dir, pore_diff_dir = \
            'dyn_data', 'pore_props', 'pore_diffs'
        # dyn_data_dir, pore_props_dir, pore_diff_dir = '', '', ''

        # self.exp_data_path = path.join(data_path, dyn_data_dir, 'dyn_data_' + sample + '.nc')
        # self.pore_props_path = path.join(data_path, pore_props_dir, 'pore_props_' + sample + '.nc')
        # self.pore_diff_path = path.join(data_path, pore_diff_dir, 'peak_diff_data_' + sample + '.nc')
        self.exp_data_path = path.join(data_path, 'dyn_data_' + sample + '.nc')
        self.pore_props_path = path.join(data_path,  'pore_props_' + sample + '.nc')
        # self.pore_diff_path = path.join(data_path, 'peak_diff_data_' + sample + '.nc')
        drive = r'\\152.88.86.87\data118'
        diff_data_path = path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives', 'processed_1200_dry_seg_aniso_sep')
        self.pore_diff_path = path.join(diff_data_path, 'peak_diff_data_' + sample + '.nc')

        self.randomize_waiting_times = rand_waiting_times
        self.pore_diff_data = None
        if path.isfile(self.pore_diff_path) and not rand_waiting_times:
            self.pore_diff_data = waitingtimes.get_diff_data(self.pore_diff_path)

        self.graph = graph
        self.data = None
        self.waiting_times = np.array([])
        self.V = None # Water volume in the network
        self.filled = None # filled nodes
        self.inlets = inlets # inlet pores
        self.R0 = None # pore resistances
        self.R_full = None # resistances when full
        self.R_inlet = R_inlet # inlet resistance

        self.radi = None
        self.heights = None
        # self.volumes = None

        if path.isfile(self.exp_data_path) and not rand_exp_data:
            print('Reading the experimental dataset at {}'.format(self.exp_data_path))
            self.data = xr.load_dataset(self.exp_data_path)
            self.generate_graph(self.data)

        self.nodes = {} # Dictionary translating labels to graph nodes. I am not sure if I quite understand, please review the correct use in lines 194&195, I need a list of the original labels
        self.label_dict = {} # Dictionary translating graph nodes to labels
        i = 0
        for node in self.graph.nodes():
            self.nodes[i] = node
            self.label_dict[node] = i
            i += 1

        # self.labels contains the list of unique identifiers for the nodes
        self.labels = np.arange(len(self.nodes))

        if path.isfile(self.pore_props_path) and not rand_pore_props:
            print('Reading the pore dataset at {}'.format(self.pore_props_path))
            pore_data = xr.load_dataset(self.pore_props_path)
            self.generate_pore_data(pore_data)
        else:
            self.generate_pore_data()

        self.generate_waiting_times()
        self.build_inlets(inlet_count)

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

        struct = cube # ball does not work as you would think (anisotropic expansion)
        # if im.ndim == 2:
            # struct = disk

        crude_pores = sp.ndimage.find_objects(im)

        # throw out None-entries (counterintuitive behavior of find_objects)
        pores = deque()
        bounding_boxes = deque()
        for pore in crude_pores:
            if pore is not None: bb = extend_bounding_box(pore, im.shape)
            if pore is not None and len(np.unique(im[bb])) > 2:
                pores.append(pore)
                bounding_boxes.append(bb)

        connections_raw = Parallel(n_jobs = self.job_count)(
            delayed(label_function)\
                (struct, im[bounding_box], label, labels, self.verbose) \
                for (bounding_box, label) in zip(bounding_boxes, labels)
        )

        # clear out empty objects
        connections = deque()
        for connection in connections_raw:
            if len(connection) > 0:
                connections.append(connection)

        return np.concatenate(connections, axis = 0)

    def generate_graph(self, exp_data):
        label_matrix = exp_data['label_matrix'].data
        labels = exp_data['label'].data
        
        
        # clean up label matrix for late spurious pixels (at fiber surfacedue to not correctable image shift )
        # raw_labels = np.unique(label_matrix)
        # for label in raw_labels[1:]:
        #     if not label in labels:
        #         label_matrix[np.where(label_matrix==label)] = 0

        if self.verbose:
            print('labels', labels)
            print('label matrix shape', label_matrix.shape)

        if self.graph is None:
            print('Generating the pore network graph from the experimental dataset')
            throats = self.extract_throat_list(label_matrix, labels)
            self.graph = nx.Graph()
            self.graph.add_edges_from(np.uint16(throats[:,:2]))
            Gcc = sorted(nx.connected_components(self.graph), key=len, reverse=True)
            self.graph = self.graph.subgraph(Gcc[0])


    def generate_pore_data(self, pore_data = None):
        if pore_data is None:
            if self.verbose:
                print('Filling the graph with random pore data')

            size = self.labels.max() + 1
            re = self.radi = np.random.rand(size)
            h0e = self.heights = np.random.rand(size)
            for i in range(size):
                re[i] /= 10**np.random.randint(5, 6)
                h0e[i] /= 10**np.random.randint(4, 5)

            return

        print('Using experimental pore data')

        
        px = pore_data.attrs['voxel']
        vx = px**3
        tmax = -1
        if self.sample in list(time_limit.keys()):
            tmax = time_limit[self.sample]
        
        relevant_labels = list(self.label_dict.keys())
        
        radi = self.radi = px*np.sqrt(pore_data['value_properties'].sel(property = 'median_area', label = relevant_labels).data/np.pi) #
        heights = self.heights = px*pore_data['value_properties'].sel(property = 'major_axis', label = relevant_labels).data #
        # volumes = self.volumes = vx*pore_data['value_properties'].sel(property = 'volume', label = relevant_labels).data
        volumes =  vx*self.data['volume'][:,tmax-10:tmax-1].sel(label = relevant_labels).median(dim='time').data#
        volumes[volumes==0] = np.median(volumes[volumes>0])
        self.volumes = volumes
        size = self.labels.max() + 1
        
        if self.data is None or self.randomize_pore_data == True:
            # corr = exp_data['sig_fit_data'].sel(sig_fit_var = 'alpha [vx]')/pore_data['value_properties'].sel(property = 'volume', label = exp_data['label'])
            # pore_data['value_properties'].sel(property = 'median_area', label = exp_data['label']) = 1/corr*pore_data['value_properties'].sel(property = 'median_area', label = exp_data['label'])
            # pore_data['value_properties'].sel(property = 'major_axis', label = exp_data['label']) = corr

            print('Initializing pore props from ECDF distribution')

#           you can use all pores even those outside the network (isolated nodes) as base for the statistical distributio here
            radi = self.radi = px*np.sqrt(pore_data['value_properties'].sel(property = 'median_area').data/np.pi)
            heights = self.heights = px*pore_data['value_properties'].sel(property = 'major_axis').data
            size = self.labels.max() + 1

            # TODO: couple radii and heights because they correlate slightly, currently the pore resulting pore volumes are too small
            # or, mix distribution functions of height, radius and volume. Something to think about ... for later ...
            ecdf_radi, ecdf_heights, ecdf_volumes = ECDF(radi), ECDF(heights), ECDF(volumes)
            seed = self.seed
            prngpore = np.random.RandomState(seed)
            prngpore2 = np.random.RandomState(seed*7+117)
            random_input1 = lambda size: prngpore.rand(size)
            random_input2 = lambda size: prngpore2.rand(size)
            # factored_input = lambda size, factor: factor*np.ones(size) # factored_input(size, 0.7)

            radi = self.radi = interp1d(ecdf_radi.y[1:], ecdf_radi.x[1:], fill_value = 'extrapolate')(random_input1(size))
            self.heights = interp1d(ecdf_heights.y[1:], ecdf_heights.x[1:], fill_value = 'extrapolate')(random_input2(size))
            volumes = self.volumes = interp1d(ecdf_volumes.y[1:], ecdf_volumes.x[1:], fill_value = 'extrapolate')(random_input2(size))    
            # self.heights = volumes/np.pi/radi**2
    def generate_waiting_times(self):
        size = self.labels.max() + 1
        data = self.pore_diff_data

        if self.randomize_waiting_times or data is None:
            print('Using random waiting times.')
            times = self.waiting_times = np.random.rand(size)
            for i in range(size):
                times[i] *= 10**np.random.randint(-1, 3)
        else:
            print('Generating waiting times from ECDF distribution')
            # print('Generating waiting times from Gamma distribution')
            self.waiting_times = waitingtimes.from_ecdf(data, len(self.labels))
            # self.waiting_times = waitingtimes.from_sigmoid_ecdf(data, len(self.labels))
            # self.waiting_times = waitingtimes.from_gamma_fit(len(self.labels))
            # TODO: get waiting times from gamma distribution again

    def build_inlets(self, amount = 5):
        inlets = np.array(self.inlets, dtype = np.int)
        if not np.any(inlets):
            self.generate_inlets(amount)
        else:
            # double-check if inlet pores are actually in the network
            temp_inlets = deque()
            print('Taking inlets from command-line arguments.')
            for inlet in inlets:
                if self.nodes[inlet] in self.graph:
                    temp_inlets.append(inlet)
            self.inlets = np.array(temp_inlets)

    # maybe TODO: Change this to start with one random inlet and some amount of distant neighbours
    def generate_inlets(self, amount):
        print('Generating {} inlets'.format(amount))
        prng = np.random.RandomState(self.seed)
        self.inlets = prng.choice(list(self.labels), size=amount, replace=False)

    def neighbour_labels(self, node):
        neighbour_nodes = self.graph.neighbors(self.nodes[node])
        neighbours = deque()
        for neighbour in neighbour_nodes:
            neighbours.append(self.label_dict[neighbour])

        return neighbours

    def outlet_resistances(self):
        """
        find your path through the filled network to calculate the inlet
        resistance imposed on the pores at the waterfront
        quick and dirty, this part makes the code slow and might even be wrong
        we have to check
        """
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
            inv_R_eff = np.float64(0)
            neighbours = self.neighbour_labels(node)

            if self.verbose:
                print('visiting node', node)

            for neighbour in neighbours:
                if neighbour in layer:
                    inv_R_eff += 1/np.float64(self.R0[neighbour] + self.R_full[neighbour]) 

            self.R0[node] += 1/inv_R_eff

            if self.filled[node] and node not in visited:
                next_layer.append(node)

            visited[node] = True

        if self.verbose:
            print('next layer', next_layer)

        return self.outlet_resistances_r(next_layer, visited)
