import random
import xarray as xr
import numpy as np
import scipy as sp
import networkx as nx

from collections import deque
from skimage.morphology import cube
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from joblib import Parallel, delayed
from os import path

import wickingpnm.waitingtimes as waitingtimes

def label_function(struct, pore_object, label, verbose = False):
    mask = pore_object == label
    connections = deque()

    if verbose:
        print('Searching around {}'.format(label))

    mask = sp.ndimage.binary_dilation(input = mask, structure = struct(3))
    neighbors = np.unique(pore_object[mask])[1:]

    for nb in neighbors:
        if nb != label:
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
        verbose = False
    ):
        self.job_count = job_count
        self.verbose = verbose

        self.data_path = data_path # Head directory for the data
        self.sample = sample # Sample name

        self.exp_data_path = path.join(data_path, 'dyn_data', 'dyn_data_' + sample + '.nc')
        self.pore_props_path = path.join(data_path, 'pore_props', 'pore_props_' + sample + '.nc')
        self.pore_diff_path = path.join(data_path, 'pore_diffs', 'peak_diff_data_' + sample + '.nc')

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

        struct = cube # FIXME: ball does not work as you would think (anisotropic expansion)
        if im.ndim == 2:
            struct = disk

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
                (struct, im[bounding_box], label, self.verbose) \
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

        if self.verbose:
            print('labels', labels)
            print('label matrix shape', label_matrix.shape)

        if self.graph is None:
            print('Generating the pore network graph from the experimental dataset')
            throats = self.extract_throat_list(label_matrix, labels)
            self.graph = nx.Graph()
            self.graph.add_edges_from(np.uint16(throats[:,:2]))

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

        else:
            print('Using experimental pore data')

            px = pore_data.attrs['voxel'].data
            radi = self.radi = px*np.sqrt(pore_data['value_properties'].sel(property = 'median_area').data/np.pi)
            heights = self.heights = px*pore_data['value_properties'].sel(property = 'major_axis').data
            size = self.labels.max() + 1
            if size < len(radi):
                print('not all pores are connected, cleaning up heights and radii')
                radi = self.radi = px*np.sqrt(pore_data['value_properties'].sel(property = 'median_area', label = list(self.label_dict.keys())).data/np.pi)
                heights = self.heights = px*pore_data['value_properties'].sel(property = 'major_axis', label = list(self.label_dict.keys())).data   
            if len(radi) < size: #it would be nice to have this as an input option, e.g. we use the experimental graph, but the properties are random
                print('Initializing pore props from ECDF distribution')
                ecdf_radi, ecdf_heights = ECDF(radi), ECDF(heights)
                self.radi = interp1d(ecdf_radi.y[1:], ecdf_radi.x[1:], fill_value = 'extrapolate')(np.random.rand(size))
                self.heights = interp1d(ecdf_heights.y[1:], ecdf_heights.x[1:], fill_value = 'extrapolate')(np.random.rand(size))

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
            self.waiting_times = waitingtimes.from_ecdf(data, len(self.labels))

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

    # TODO: Change this to start with one random inlet and some amount of distant neighbours
    def generate_inlets(self, amount):
        print('Generating {} inlets'.format(amount))
        self.inlets = random.sample(list(self.labels), amount)

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
                    inv_R_eff += 1/np.float64(self.R0[neighbour] + self.R_full[neighbour]) #<- you sure about this? you add a resistance to an inverse resistance

            self.R0[node] += 1/inv_R_eff

            if self.filled[node] and node not in visited:
                next_layer.append(node)

            visited[node] = True

        if self.verbose:
            print('next layer', next_layer)

        return self.outlet_resistances_r(next_layer, visited)
