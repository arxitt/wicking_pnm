import xarray as xr
import numpy as np
import scipy as sp
import networkx as nx

from collections import deque
from joblib import Parallel, delayed
from skimage.morphology import cube

from wickingpnm.model.network import PNM

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

class ExpPNM(PNM):
    def __init__(self, stats_path, exp_data_path, pore_data_path, **kwargs):
        super().__init__(stats_path, **kwargs)

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

        # define waiting times and inlets
        self.generate_waiting_times()
        self.build_inlets()

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
                (struct, im[bounding_box], label, self.verbose) \
                for (bounding_box, label) in zip(bounding_boxes, labels)
        )

        # clear out empty objects
        connections = deque()
        for connection in connections_raw:
            if len(connection) > 0:
                connections.append(connection)

        return np.concatenate(connections, axis = 0)
